from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.confidence import (  # noqa: E402
    ACTION_CONFIDENCE_FIELD_NAMES,
    ActionConfidenceCalibrator,
    ActionConfidenceConfig,
    save_action_confidence_npz,
)
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
from rl_quant.data_sources.polygon_second_aggs import (  # noqa: E402
    PolygonSecondAggConfig,
    available_timestamp_ms,
    iso_to_timestamp_ms,
    load_manifest,
    timestamp_ms_to_iso,
    validate_manifest,
)
from rl_quant.data_sources.polygon_stock_covariates import (  # noqa: E402
    normalize_raw_covariate_record,
    regular_session_open_ms_after_date,
)
from rl_quant.features.stock_covariates import (  # noqa: E402
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
from rl_quant.features.news_llm import (  # noqa: E402
    NEWS_LLM_AGGREGATE_FEATURE_NAMES,
    NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
    NEWS_LLM_EXTRACT_SCHEMA_VERSION,
    aggregate_news_llm_features_for_symbol,
    build_action_news_llm_tensor,
    build_deterministic_news_llm_rows,
    build_news_article_rows,
    write_news_llm_feature_outputs,
)
from rl_quant.features.stock_second_context import (  # noqa: E402
    StockSecondContextConfig,
    build_second_context_payload,
    regular_session_decision_grid_ms,
    save_second_context_payload,
    validate_second_context_payload,
)
from rl_quant.research_protocol import stable_json_hash  # noqa: E402
from rl_quant.second_context_transformer import (  # noqa: E402
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
    @staticmethod
    def _write_bar_csv(path: Path, rows: list[tuple[str, str]]) -> None:
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
            for timestamp, close in rows:
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

    def test_direct_hourly_builder_marks_legacy_protocol_non_reportable(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_dir = root / "stocks"
            etf_dir = root / "etfs"
            output_dir = root / "out"
            stock_dir.mkdir()
            etf_dir.mkdir()
            rows = [
                (f"2026-01-02T14:{30 + offset:02d}:00+00:00", str(100 + offset))
                for offset in range(11)
            ]
            self._write_bar_csv(stock_dir / "AAA_1m.csv", rows)
            self._write_bar_csv(etf_dir / "QQQ_1m.csv", rows)
            stock_universe = root / "stocks.csv"
            etf_universe = root / "etfs.csv"
            stock_universe.write_text("symbol\nAAA\n")
            etf_universe.write_text("symbol\nQQQ\n")
            old_argv = sys.argv
            sys.argv = [
                "build_hourly_transformer_dataset.py",
                "--stock-bar-dir",
                str(stock_dir),
                "--etf-bar-dir",
                str(etf_dir),
                "--stock-universe",
                str(stock_universe),
                "--etf-universe",
                str(etf_universe),
                "--output-dir",
                str(output_dir),
                "--dataset-file-name",
                "dataset.pt",
                "--bar-interval",
                "1m",
                "--start",
                "2026-01-02T00:00:00+00:00",
                "--end-exclusive",
                "2026-01-03T00:00:00+00:00",
                "--stock-limit",
                "1",
                "--action-count",
                "1",
                "--min-active-stock-fraction",
                "1.0",
                "--universe-selection-date",
                "2026-01-01T00:00:00+00:00",
            ]
            try:
                module.main()
            finally:
                sys.argv = old_argv

            payload = torch.load(output_dir / "dataset.pt", map_location="cpu", weights_only=True)
            manifest = json.loads((output_dir / "dataset_manifest.json").read_text())

        self.assertFalse(payload["dataset_reportable"])
        self.assertIn("legacy_hourly_all_labels_required_protocol", payload["dataset_reportability_errors"])
        self.assertIn("rows_filtered_by_future_label_availability", payload["dataset_reportability_errors"])
        self.assertTrue(torch.equal(payload["decision_action_valid_mask"], payload["action_valid_mask"]))
        self.assertTrue(torch.equal(payload["label_valid_mask"], payload["action_label_valid_mask"]))
        self.assertFalse(manifest["reportable"])
        self.assertIn("legacy_hourly_all_labels_required_protocol", manifest["reportability_errors"])


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

        # Turnover budget (daily switch cap) exhausted: a switch to cash also consumes
        # turnover, so the cap legitimately restricts to holding the current action only.
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

        # Episode turnover budget exhausted: restrict to the current action (cash switch
        # also consumes turnover and is not exempt from the budget).
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

        result = evaluate_minute_to_hour_policy(
            split,
            PickActionPolicy(),
            device=torch.device("cpu"),
            initial_action=0,
            constraints=TradingConstraintConfig(one_way_cost_bps=0.0),
            capture_rollout=True,
        )

        self.assertEqual([row["asset"] for row in result.rollout_records], ["CASH", "QQQ"])
        self.assertFalse(result.evaluation_reportable)
        self.assertEqual(result.selectable_missing_label_count, 1)
        self.assertEqual(result.requested_action_missing_label_count, 1)
        self.assertEqual(result.executed_action_missing_label_count, 0)
        self.assertEqual(result.policy_unscorable_rows, 1)
        self.assertIn("requested_actions_with_missing_reward_labels", result.reportability_errors)
        self.assertEqual([row["requested_asset"] for row in result.rollout_records], ["QQQ", "QQQ"])
        self.assertEqual([row["executed_asset"] for row in result.rollout_records], ["CASH", "QQQ"])

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

        self.assertEqual(selected, ["2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"])
        self.assertEqual(module.partition_selection_reportability_errors(args), [])

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

        self.assertEqual(selected, ["2026-01-01_to_2026-01-04", "2026-06-01_to_2026-06-04"])
        self.assertEqual(module.partition_selection_reportability_errors(args), [])

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
        self.assertEqual(policy["partition_selection_reportability_errors"], ["non_latest_partition_selection"])
        self.assertIn("non_latest_partition_selection", policy["reportability_errors"])
        self.assertFalse(reportable)
        self.assertIn("smoke_row_based_split", errors)
        self.assertIn("non_latest_partition_selection", errors)
        self.assertFalse(evaluator_error_reportable)
        self.assertEqual(evaluator_errors, ["requested_actions_with_missing_reward_labels"])


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

    def test_direct_hourly_split_orders_mixed_timezones_by_utc_instant(self) -> None:
        hourly = __import__("rl_quant.hourly_transformer", fromlist=["_build_split"])
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
        hourly = __import__("rl_quant.hourly_transformer", fromlist=["_build_split"])
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

    def test_validate_reportable_summary_flags_missing_dataset_manifest_file(self) -> None:
        errors = validate_reportable_summary(
            {
                "dataset_manifest": {"manifest_available": False},
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

        self.assertEqual(errors, ["dataset_manifest_file_missing"])


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_local_absolute_path_literals(self) -> None:
        forbidden = [
            "/" + part
            for part in [
                "home/yding1995",
                "tmp/",
                "mnt/",
                "root/",
                "Users/",
                "path/to",
            ]
        ]
        excluded_parts = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
        excluded_names = {"test_correctness.py"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or excluded_parts.intersection(path.parts) or path.name in excluded_names:
                continue
            text = path.read_text(errors="ignore")
            for needle in forbidden:
                self.assertNotIn(needle, text, f"Local absolute path literal {needle!r} found in {path}")

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

        missing = DatasetManifest.from_dict(manifest.to_dict())
        missing.universe_selection_date = None
        with self.assertRaisesRegex(ResearchProtocolError, "universe_selection_date is required"):
            missing.validate()

    def test_universe_selection_date_resolver_uses_latest_universe_file_date(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        args = module.parse_args(
            [
                "--stock-universe",
                "data/top_us_volume_stocks_2026-06-14.csv",
                "--etf-universe",
                "data/top_us_volume_etfs_2026-06-13.csv",
            ]
        )

        self.assertEqual(module.resolve_universe_selection_date(args), "2026-06-14T00:00:00+00:00")

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

    def _reportable_manifest(self) -> ModelManifest:
        protocol = EvaluationProtocol(
            name="holdout",
            train_start=None,
            train_end="2026-01-31T00:00:00+00:00",
            val_end="2026-02-28T00:00:00+00:00",
            test_start="2026-03-01T00:00:00+00:00",
            test_end="2026-03-31T00:00:00+00:00",
            benchmark_names=["CASH"],
        )
        return ModelManifest(
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
            baseline_results=[BaselineResult("CASH", 0.0, None, 0.0)],
            cost_stress_results=[StressTestResult("2x_cost", "cost", "multiplier", 2.0, 0.0, None, 0.0)],
            frequency_stress_results=[
                StressTestResult("min_hold_2", "frequency", "min_hold_bars", 2.0, 0.0, None, 0.0)
            ],
        )

    def test_model_manifest_rejects_test_split_selection(self) -> None:
        manifest = self._reportable_manifest()
        manifest.validate_reportable()
        manifest.selected_by = "test_total_return"
        with self.assertRaisesRegex(ResearchProtocolError, "selected_by must reference validation"):
            manifest.validate_reportable()

    def test_model_manifest_rejects_unproduced_benchmark(self) -> None:
        manifest = self._reportable_manifest()
        manifest.validation_protocol = EvaluationProtocol(
            name="holdout",
            train_start=None,
            train_end="2026-01-31T00:00:00+00:00",
            val_end="2026-02-28T00:00:00+00:00",
            test_start="2026-03-01T00:00:00+00:00",
            test_end="2026-03-31T00:00:00+00:00",
            benchmark_names=["CASH", "RandomSameTurnover"],
        )
        with self.assertRaisesRegex(ResearchProtocolError, "missing from baseline_results"):
            manifest.validate_reportable()

    def test_default_benchmark_registry_matches_action_universe(self) -> None:
        benchmarks = default_benchmark_registry(["CASH", "QQQ", "SPY"])

        self.assertIn("CASH", benchmarks)
        self.assertIn("BuyAndHold_QQQ", benchmarks)
        self.assertIn("RandomSameTurnover", benchmarks)


class CoreAndFixRegressionTests(unittest.TestCase):
    """Tests added to close prior coverage gaps and pin the correctness fixes."""

    def test_replay_buffer_wraparound_and_over_capacity(self) -> None:
        from rl_quant.core import TensorReplayBuffer

        buf = TensorReplayBuffer(capacity=4, device=torch.device("cpu"), fields={"x": torch.long})
        buf.add(x=torch.tensor([0, 1, 2]))
        self.assertEqual(buf.size, 3)
        buf.add(x=torch.tensor([3, 4]))  # wraps past capacity: keeps the 4 most recent
        self.assertEqual(buf.size, 4)
        self.assertEqual(sorted(buf.storage["x"].tolist()), [1, 2, 3, 4])

        big = TensorReplayBuffer(capacity=3, device=torch.device("cpu"), fields={"x": torch.long})
        big.add(x=torch.tensor([10, 11, 12, 13, 14]))  # single add larger than capacity
        self.assertEqual(big.size, 3)
        self.assertEqual(sorted(big.storage["x"].tolist()), [12, 13, 14])

    def test_annualized_sharpe_and_drawdowns(self) -> None:
        from rl_quant.core import absolute_max_drawdown, annualized_sharpe, fractional_max_drawdown

        self.assertIsNone(annualized_sharpe([1.0]))
        self.assertIsNone(annualized_sharpe([0.01, 0.01]))  # zero sigma -> None
        self.assertIsNotNone(annualized_sharpe([0.01, -0.01, 0.02, -0.02]))
        self.assertAlmostEqual(fractional_max_drawdown([100.0, 50.0, 75.0]), -0.5, places=6)
        self.assertAlmostEqual(absolute_max_drawdown([100.0, 50.0, 75.0]), 50.0, places=6)

    def test_nbbo_builder_midpoint_spread_and_crossed_flag(self) -> None:
        from rl_quant.quote_utils import NbboBuilder

        builder = NbboBuilder()
        snap = builder.update(exchange="A", bid=100.0, bid_size_lots=2, ask=100.2, ask_size_lots=3, timestamp_ns=1)
        self.assertIsNotNone(snap)
        self.assertAlmostEqual(snap.mid, 100.1, places=6)
        self.assertAlmostEqual(snap.spread, 0.2, places=6)
        self.assertFalse(snap.crossed)
        self.assertFalse(snap.locked)
        # A second venue bidding above the best ask crosses the book; it must be FLAGGED.
        crossed = builder.update(exchange="B", bid=100.5, bid_size_lots=1, ask=100.6, ask_size_lots=1, timestamp_ns=2)
        self.assertTrue(crossed.crossed)
        self.assertLess(crossed.spread, 0.0)

    def test_causal_mask_disallows_future_positions(self) -> None:
        from rl_quant.hourly_transformer import CausalTransformerQNetwork

        net = CausalTransformerQNetwork(feature_dim=3, lookback=4, action_count=3)
        mask = net._causal_mask(4, torch.device("cpu"))
        for i in range(4):
            for j in range(4):
                if j <= i:
                    self.assertEqual(float(mask[i, j].item()), 0.0)  # self + past allowed
                else:
                    self.assertLess(float(mask[i, j].item()), -1e30)  # future disallowed

    def test_calibrator_recovers_known_residual_std_and_flags_in_sample(self) -> None:
        rows = 100
        amplitude = 0.05
        series = torch.tensor([amplitude if i % 2 == 0 else -amplitude for i in range(rows)])
        returns = torch.stack([series, series], dim=1)
        q_values = torch.zeros((rows, 2))
        valid = torch.ones((rows, 2), dtype=torch.bool)
        config = ActionConfidenceConfig(q_value_scale=1.0, min_calibration_rows=1, ood_penalty=False)
        calibrator = ActionConfidenceCalibrator(config).fit(q_values, returns, valid)
        self.assertAlmostEqual(calibrator.metrics["global_residual_std"], amplitude, places=6)
        self.assertTrue(calibrator.metrics["in_sample_optimistic"])

    def test_p_beats_cash_is_nan_for_cash_self_comparison(self) -> None:
        rows = 40
        amplitude = 0.05
        series = torch.tensor([amplitude if i % 2 == 0 else -amplitude for i in range(rows)])
        returns = torch.stack([torch.zeros(rows), series], dim=1)
        q_values = torch.zeros((rows, 2))
        valid = torch.ones((rows, 2), dtype=torch.bool)
        config = ActionConfidenceConfig(q_value_scale=1.0, min_calibration_rows=1, ood_penalty=False)
        calibrator = ActionConfidenceCalibrator(config).fit(q_values, returns, valid)
        out = calibrator.predict(q_values, valid)
        self.assertTrue(torch.isnan(out.p_beats_cash[:, 0]).all())  # CASH vs CASH undefined
        self.assertTrue(torch.isfinite(out.p_beats_cash[:, 1]).all())

    def test_empty_market_block_marks_all_symbols_missing(self) -> None:
        from rl_quant.features.stock_second_context import (
            MARKET_CONTEXT_FEATURE_NAMES,
            _block_market_features,
        )

        features, valid = _block_market_features(
            {}, block_start_ms=0, block_end_ms=1_000, total_symbols=10, min_active_symbols=5
        )
        self.assertFalse(valid)
        missing_index = MARKET_CONTEXT_FEATURE_NAMES.index("missing_symbol_fraction")
        self.assertEqual(features[missing_index], 1.0)

    def test_precomputed_news_requires_model_availability_for_real_model(self) -> None:
        module = load_script("build_news_llm_features")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"ticker": "QQQ", "model_available_timestamp_ms": 0}) + "\n")
            _, errors = module.read_precomputed_rows(path, require_model_availability=True)
        self.assertTrue(any("model_available_timestamp_ms must be > 0" in error for error in errors))

    def test_model_manifest_adapter_accepts_trainer_schema(self) -> None:
        module = load_script("validate_research_protocol")
        trainer_manifest = {
            "model_id": "demo",
            "created_at_utc": "2026-06-14T00:00:00+00:00",
            "algorithm": "DoubleDQN",
            "encoder": "CausalTransformer",
            "training_dataset": "/data/demo.pt",
            "hyperparameters_hash": "abc",
            "selected_by": "best_validation_total_return",
            "feature_names_hash": "fh",
            "action_names_hash": "ah",
            "validation_protocol": {
                "train_end": "2026-01-31T00:00:00+00:00",
                "val_end": "2026-02-28T00:00:00+00:00",
                "test_start": "2026-03-01T00:00:00+00:00",
                "test_end": "2026-03-31T00:00:00+00:00",
                "purge_rule": "chronological_no_overlap",
            },
            "baseline_results": [
                {"name": "CASH", "total_return": 0.0, "annualized_sharpe": None, "max_drawdown": 0.0, "total_switches": 0}
            ],
            "cost_stress_results": [{"name": "2x", "kind": "fixed_rollout", "cost_bps": 2.0, "total_return": 0.0}],
            "frequency_stress_results": [{"name": "min_hold_2", "kind": "frequency", "total_return": 0.0}],
            "action_metadata_hash": "should_be_dropped",
            "constraints": {"min_hold_bars": 1},
        }
        manifest = module.load_model_manifest(trainer_manifest)
        manifest.validate_reportable()
        self.assertEqual(manifest.model_id, "demo")
        self.assertIn("CASH", manifest.validation_protocol.benchmark_names)

    def test_ecology_helpers_softmax_and_drawdown_semantics(self) -> None:
        module = load_script("learn_market_ecological_attention")
        uniform = module.softmax([0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(uniform), 1.0, places=6)
        self.assertTrue(all(abs(weight - 1.0 / 3.0) < 1e-6 for weight in uniform))
        skewed = module.softmax([10.0, 0.0])
        self.assertGreater(skewed[0], skewed[1])
        # trailing_returns over a small series equals the hand-computed pct changes.
        series = {"2026-01-01": (100.0, 0.0), "2026-01-02": (110.0, 0.0), "2026-01-03": (104.5, 0.0)}
        dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
        rets = module.trailing_returns(series, dates, 2, 3)
        self.assertAlmostEqual(rets[-1], 104.5 / 110.0 - 1.0, places=6)
        # trailing_drawdown is the CURRENT drawdown from the window peak (NOT max drawdown):
        # peak is 110, current is 104.5 -> 104.5/110 - 1, even though an intermediate max-DD differs.
        drawdown = module.trailing_drawdown(series, dates, 2, 3)
        self.assertAlmostEqual(drawdown, 104.5 / 110.0 - 1.0, places=6)

    @staticmethod
    def _valid_news_llm_row(**overrides):
        from rl_quant.features.news_llm import NEWS_LLM_ARTICLE_TICKER_FIELDS

        ts = 1_700_000_000_000
        row = {field: 0.0 for field in NEWS_LLM_ARTICLE_TICKER_FIELDS}
        row.update(
            {
                "article_id": "a1",
                "ticker": "QQQ",
                "published_utc": "2026-01-05T15:00:00+00:00",
                "published_timestamp_ms": ts,
                "source_available_timestamp_ms": ts,
                "llm_feature_available_timestamp_ms": ts,
                "model_available_timestamp_ms": 0,
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
                "confidence": 0.8,
                "llm_valid": True,
                "llm_model_id": "m",
                "llm_prompt_hash": "p",
                "llm_schema_version": NEWS_LLM_EXTRACT_SCHEMA_VERSION,
                "llm_schema_hash": NEWS_LLM_ARTICLE_TICKER_SCHEMA_HASH,
                "extractor_provider": "local_transformers",
                "extractor_temperature": 0.0,
                "extractor_no_retrieval": True,
                "model_training_cutoff_utc": "unknown",
                "article_weight": 1.0,
                "ticker_count": 1.0,
            }
        )
        row.update(overrides)
        return row

    def test_news_llm_validation_rejects_bad_rows(self) -> None:
        from rl_quant.features.news_llm import validate_news_llm_rows

        self.assertEqual(validate_news_llm_rows([self._valid_news_llm_row()]), [])
        self.assertTrue(
            any("sentiment_score" in e for e in validate_news_llm_rows([self._valid_news_llm_row(sentiment_score=2.0)]))
        )
        self.assertTrue(
            any("confidence" in e for e in validate_news_llm_rows([self._valid_news_llm_row(confidence=float("nan"))]))
        )
        self.assertTrue(
            any(
                "source availability" in e
                for e in validate_news_llm_rows([self._valid_news_llm_row(source_available_timestamp_ms=2_000_000_000_000)])
            )
        )
        self.assertTrue(
            any(
                "duplicate" in e
                for e in validate_news_llm_rows([self._valid_news_llm_row(), self._valid_news_llm_row()])
            )
        )

    def test_news_llm_content_hash_changes_with_feature_value(self) -> None:
        if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pandas/pyarrow required for feature-table hashing test")
        common = dict(
            article_manifest=None,
            model_id="m",
            model_available_timestamp_ms=0,
            model_training_cutoff_utc="unknown",
            provider="local_transformers",
        )
        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            manifest_a = write_news_llm_feature_outputs(rows=[self._valid_news_llm_row()], output_root=Path(dir_a), **common)
            manifest_b = write_news_llm_feature_outputs(
                rows=[self._valid_news_llm_row(sentiment_score=-0.5)], output_root=Path(dir_b), **common
            )
        self.assertTrue(manifest_a["reportable"])
        # The content hash MUST change when a feature value changes ...
        self.assertNotEqual(manifest_a["feature_table_content_hash"], manifest_b["feature_table_content_hash"])
        # ... while the legacy identity-only hash does not (the exact gap the content hash closes).
        self.assertEqual(manifest_a["feature_table_hash"], manifest_b["feature_table_hash"])

    def test_news_llm_mixed_provenance_is_nonreportable_and_quarantined(self) -> None:
        if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pandas/pyarrow required for feature-table write test")
        rows = [
            self._valid_news_llm_row(article_id="a1", llm_prompt_hash="p1"),
            self._valid_news_llm_row(article_id="a2", llm_prompt_hash="p2"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_news_llm_feature_outputs(
                rows=rows,
                output_root=Path(directory),
                article_manifest=None,
                model_id="m",
                model_available_timestamp_ms=0,
                model_training_cutoff_utc="unknown",
                provider="local_transformers",
            )
            self.assertFalse(manifest["reportable"])
            self.assertTrue(manifest["mixed_provenance"])
            # The canonical (reportable) path must NOT be materialized; the table is quarantined.
            self.assertFalse((Path(directory) / "news_article_ticker_llm.parquet").exists())
            self.assertTrue((Path(directory) / "news_article_ticker_llm.nonreportable.parquet").exists())

    def test_news_llm_validation_rejects_nonzero_temperature(self) -> None:
        from rl_quant.features.news_llm import validate_news_llm_rows

        # A sampled (nonzero-temperature) extraction must be non-reportable, so the generator
        # recording the ACTUAL temperature makes such rows fail validation rather than masquerade
        # as deterministic.
        errors = validate_news_llm_rows([self._valid_news_llm_row(extractor_temperature=0.7)])
        self.assertTrue(any("extractor_temperature" in error for error in errors))
        # A boolean in a continuous-score field is rejected (not silently coerced to 0/1).
        bad_bool = validate_news_llm_rows([self._valid_news_llm_row(confidence=True)])
        self.assertTrue(any("confidence" in error for error in bad_bool))

    def test_news_llm_nonreportable_write_quarantines_and_reader_skips(self) -> None:
        if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pandas/pyarrow required for feature-table write test")
        from rl_quant.features.news_llm import read_news_llm_rows

        common = dict(
            article_manifest=None,
            model_id="m",
            model_available_timestamp_ms=0,
            model_training_cutoff_utc="unknown",
            provider="local_transformers",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # First a valid build -> canonical table exists and is readable.
            write_news_llm_feature_outputs(rows=[self._valid_news_llm_row()], output_root=root, **common)
            self.assertTrue((root / "news_article_ticker_llm.parquet").exists())
            self.assertEqual(len(read_news_llm_rows(root)), 1)
            # Then a non-reportable (mixed-provenance) build -> canonical removed, reader fails closed.
            write_news_llm_feature_outputs(
                rows=[
                    self._valid_news_llm_row(article_id="a1", llm_prompt_hash="p1"),
                    self._valid_news_llm_row(article_id="a2", llm_prompt_hash="p2"),
                ],
                output_root=root,
                **common,
            )
            self.assertFalse((root / "news_article_ticker_llm.parquet").exists())
            # Fail closed by default on a non-reportable manifest (don't confuse it with zero news).
            with self.assertRaises(ValueError):
                read_news_llm_rows(root)
            self.assertEqual(len(read_news_llm_rows(root, allow_nonreportable=True)), 2)

    def test_news_llm_aggregate_mean_masked_when_no_news_but_count_valid(self) -> None:
        from rl_quant.features.news_llm import build_deterministic_news_llm_rows

        article = {
            "article_id": "article-1",
            "published_utc": "2026-01-05T15:00:00+00:00",
            "published_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "source_available_timestamp_ms": iso_to_timestamp_ms("2026-01-05T15:00:00+00:00"),
            "title": "QQQ outlook",
            "description": "Mixed update.",
            "tickers_json": json.dumps(["QQQ"]),
            "primary_ticker": "QQQ",
        }
        rows = build_deterministic_news_llm_rows(
            [article], model_available_timestamp_ms=0, vendor_latency_seconds=300, processing_latency_seconds=60
        )
        # A decision BEFORE the row becomes available: the window is empty.
        before = iso_to_timestamp_ms("2026-01-05T15:05:59+00:00")
        count_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("log1p_llm_weighted_news_count_1h")
        sentiment_idx = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index("llm_net_sentiment_1d")
        _values, mask, _available, _age = aggregate_news_llm_features_for_symbol(
            rows=rows, decision_ms=before, source_available=True
        )
        # Count feature is valid (0 = "no news"); the mean/sentiment feature is masked invalid so a
        # 0.0 cannot be confused with neutral sentiment.
        self.assertTrue(mask[count_idx])
        self.assertFalse(mask[sentiment_idx])

    def test_news_llm_validation_rejects_invalid_time_horizon(self) -> None:
        from rl_quant.features.news_llm import validate_news_llm_rows

        errors = validate_news_llm_rows([self._valid_news_llm_row(time_horizon="next_decade")])
        self.assertTrue(any("time_horizon" in error for error in errors))
        self.assertEqual(validate_news_llm_rows([self._valid_news_llm_row(time_horizon="days_to_weeks")]), [])

    def test_second_bar_execution_latency_default_and_explicit_floor(self) -> None:
        module = load_script("build_hourly_from_minute_context_dataset")
        # Second source data auto-sets execution latency to one bar when left at the 0 default.
        auto = module.parse_args(["--source-bar-interval", "1s"])
        self.assertEqual(auto.execution_latency_ms, module.DEFAULT_SECOND_BAR_LATENCY_MS)
        # An explicit sub-one-bar execution latency for second data is rejected, not silently used.
        with self.assertRaises(ValueError):
            module.parse_args(["--source-bar-interval", "1s", "--execution-latency-ms", "500"])

    def test_masked_mean_std_ignores_nan_in_masked_positions(self) -> None:
        from rl_quant.minute_to_hour_transformer import _masked_mean_std

        # A NaN in a MASKED-OUT position must not poison the channel statistics (NaN * 0 == NaN).
        # Two valid observations (3.0, 5.0) -> mean 4.0, NaN at the masked position ignored.
        features = torch.tensor([[[[3.0], [5.0], [float("nan")]]]])
        mask = torch.tensor([[[True, True, False]]])
        mean, std = _masked_mean_std(features, mask)
        self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(std).all())
        self.assertAlmostEqual(float(mean.item()), 4.0, places=6)

    def test_masked_mean_std_leaves_sparse_channel_unnormalized(self) -> None:
        from rl_quant.minute_to_hour_transformer import _masked_mean_std

        # Fewer than two valid observations -> mean 0, std 1 (no amplification by a near-zero std).
        features = torch.tensor([[[[3.0], [float("nan")]]]])
        mask = torch.tensor([[[True, False]]])
        mean, std = _masked_mean_std(features, mask)
        self.assertEqual(float(mean.item()), 0.0)
        self.assertEqual(float(std.item()), 1.0)

    def test_news_article_source_latency_is_applied(self) -> None:
        from rl_quant.features.news_llm import _raw_article_row

        payload = {"article_id": "a1", "published_utc": "2026-01-05T15:00:00+00:00", "title": "QQQ update"}
        row = _raw_article_row("QQQ", payload, 0, source_latency_seconds=300)
        self.assertEqual(
            int(row["source_available_timestamp_ms"]), int(row["published_timestamp_ms"]) + 300 * 1000
        )
        baseline = _raw_article_row("QQQ", payload, 0)
        self.assertEqual(int(baseline["source_available_timestamp_ms"]), int(baseline["published_timestamp_ms"]))

    def test_partition_trainer_defaults_to_fail_and_strict(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        args = module.parse_args([])
        self.assertEqual(args.insufficient_split_policy, "fail")
        self.assertEqual(args.reportability_policy, "strict")
        self.assertFalse(args.smoke)

    def test_news_llm_writer_records_generation_diagnostics(self) -> None:
        if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pandas/pyarrow required for feature-table write test")
        diagnostics = {"parse_error_fraction": 0.0, "invalid_llm_row_fraction": 0.0, "rows_written_this_run": 1}
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_news_llm_feature_outputs(
                rows=[self._valid_news_llm_row()],
                output_root=Path(directory),
                article_manifest=None,
                model_id="m",
                model_available_timestamp_ms=0,
                model_training_cutoff_utc="unknown",
                provider="local_transformers",
                generation_diagnostics=diagnostics,
            )
        self.assertEqual(manifest["generation_diagnostics"], diagnostics)

    def test_news_article_table_rejects_negative_source_latency(self) -> None:
        module = load_script("build_news_article_table")
        with self.assertRaises(SystemExit):
            module.main(["--source-latency-seconds", "-1"])

    def test_partition_latest_available_and_non_latest_selection(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("2026-01-01", "2026-01-02", "2026-01-03"):
                part = root / label
                part.mkdir(parents=True)
                (part / "hour_from_second_dataset.pt").write_bytes(b"")
            base = SimpleNamespace(
                partitions_root=root,
                dataset_file_name="hour_from_second_dataset.pt",
                start_partition=None,
                end_partition=None,
                max_partitions=0,
                partition_selection="latest",
            )
            self.assertEqual(module.latest_available_partition_label(base), "2026-01-03")
            # Selecting the earliest partition makes the final selected partition NOT the latest
            # available -- the condition the strict latest-period guard rejects.
            earliest = SimpleNamespace(**{**vars(base), "max_partitions": 1, "partition_selection": "earliest"})
            selected = module.partition_paths(earliest)
            self.assertEqual(selected[-1].parent.name, "2026-01-01")
            self.assertNotEqual(selected[-1].parent.name, module.latest_available_partition_label(earliest))

    def test_action_feature_normalizer_one_valid_value_gets_mean_zero_std_one(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_action_feature_mean_std"])
        # Channel "one" has a single mask-true value (5.0; the other entry is masked out); channel "two"
        # has two mask-true values (2.0, 6.0). (B=2, S=1, F=4) with interleaved value/mask channels.
        features = torch.tensor(
            [
                [[5.0, 1.0, 2.0, 1.0]],
                [[99.0, 0.0, 6.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        names = [
            "stock_covariates_v1.one",
            "stock_covariates_v1_mask.one",
            "stock_covariates_v1.two",
            "stock_covariates_v1_mask.two",
        ]
        mean, std = module._action_feature_mean_std(features, names)
        # Single valid observation -> unnormalized (mean 0, std 1), not amplified by a 1e-6 std.
        self.assertEqual(float(mean[0].item()), 0.0)
        self.assertEqual(float(std[0].item()), 1.0)
        # Two valid observations -> real statistics (mean 4.0, population std 2.0).
        self.assertAlmostEqual(float(mean[2].item()), 4.0, places=5)
        self.assertAlmostEqual(float(std[2].item()), 2.0, places=5)

    def test_strict_latest_partition_violations_detects_calendar_latest_and_truncation(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        fn = module.strict_latest_partition_violations
        labels = ["2026-01-01", "2026-01-02", "2026-01-03"]
        # Full selection ending at the latest available partition is admissible.
        self.assertEqual(fn(selected_labels=labels, all_available_labels=labels, allow_truncated_training_history=False), [])
        # Regex-matching but impossible calendar date is rejected.
        calendar = fn(
            selected_labels=["2026-99-99"],
            all_available_labels=["2026-01-01", "2026-99-99"],
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("invalid labels" in violation for violation in calendar))
        # Final selected partition is not the latest available -> violation.
        not_latest = fn(selected_labels=["2026-01-01"], all_available_labels=labels, allow_truncated_training_history=True)
        self.assertTrue(any("not the latest available" in violation for violation in not_latest))
        # Earliest selected partition is not the earliest available -> truncated history violation...
        truncated = fn(
            selected_labels=["2026-01-02", "2026-01-03"],
            all_available_labels=labels,
            allow_truncated_training_history=False,
        )
        self.assertTrue(any("silently excluded" in violation for violation in truncated))
        # ...unless explicitly allowed (and the final partition is still the latest available).
        self.assertEqual(
            fn(selected_labels=["2026-01-02", "2026-01-03"], all_available_labels=labels, allow_truncated_training_history=True),
            [],
        )

    def test_official_test_block_summarizes_latest_partition(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        records = [
            {"partition": "2026-01-01", "ordinal": 1, "status": "ok", "is_official_latest_test": False, "test_total_return": 0.01},
            {
                "partition": "2026-01-02",
                "ordinal": 2,
                "status": "ok",
                "is_official_latest_test": True,
                "evaluation_reportable": True,
                "reportability_errors": [],
                "test_total_return": 0.05,
                "test_switches": 3,
                "test_order_legs": 4,
                "val_total_return": 0.02,
            },
        ]
        block = module.official_test_block(records, final_test_is_latest_available=True)
        self.assertEqual(block["partition"], "2026-01-02")
        self.assertTrue(block["is_latest_available"])
        self.assertTrue(block["reportable"])
        self.assertEqual(block["test_total_return"], 0.05)
        self.assertEqual(block["test_switches"], 3)
        # No official partition yet (e.g. before the latest partition completes) -> None.
        self.assertIsNone(module.official_test_block([records[0]], True))

    def test_recency_weights_decrease_with_age_and_keep_min_weight(self) -> None:
        from rl_quant.minute_to_hour_transformer import _timestamp_to_epoch_ms, compute_recency_weights

        val_start_ms = _timestamp_to_epoch_ms("2026-06-01T13:30:00+00:00")
        timestamps = [
            "2026-01-01T14:30:00+00:00",  # ~151 days before validation
            "2026-04-01T14:30:00+00:00",  # ~61 days before validation
            "2026-05-31T14:30:00+00:00",  # ~1 day before validation
        ]
        weights = compute_recency_weights(
            timestamps, val_start_ms, mode="exponential", half_life_days=60.0, min_weight=0.05
        )
        # Older rows get strictly smaller weights, never below min_weight, never above 1.0.
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))
        self.assertGreaterEqual(float(weights[0]), 0.05)
        self.assertLessEqual(float(weights[2]), 1.0)
        self.assertGreater(float(weights[2]), 0.95)  # ~1 day old -> near full weight

    def test_recency_weights_anchor_validation_start_not_test_end(self) -> None:
        from rl_quant.minute_to_hour_transformer import _timestamp_to_epoch_ms, compute_recency_weights

        val_start = "2026-06-01T13:30:00+00:00"
        val_start_ms = _timestamp_to_epoch_ms(val_start)
        # A row exactly at the validation start has age 0 -> weight 1.0 (anchored to validation, not test).
        anchor = compute_recency_weights([val_start], val_start_ms, mode="exponential", half_life_days=60.0, min_weight=0.05)
        self.assertAlmostEqual(float(anchor[0]), 1.0, places=6)
        # A row AFTER the validation start (a test-era timestamp) clamps to age 0 -> weight 1.0; the
        # function never produces >1 (negative-age amplification) and only consumes the rows it is given.
        later = compute_recency_weights(
            ["2026-07-01T13:30:00+00:00"], val_start_ms, mode="exponential", half_life_days=60.0, min_weight=0.05
        )
        self.assertAlmostEqual(float(later[0]), 1.0, places=6)

    def test_recency_weights_mode_none_is_uniform_and_weighted_mean_equals_mean(self) -> None:
        from rl_quant.minute_to_hour_transformer import compute_recency_weights

        weights = compute_recency_weights(
            ["2026-01-01T14:30:00+00:00", "2026-05-31T14:30:00+00:00"],
            0,
            mode="none",
            half_life_days=60.0,
            min_weight=0.05,
        )
        self.assertTrue(torch.equal(weights, torch.ones(2)))
        # No-regression guarantee: a uniform-weighted mean equals an unweighted mean exactly.
        per_sample = torch.tensor([0.2, 0.4, 0.9, 1.3])
        ones = torch.ones(4)
        weighted = (per_sample * ones).sum() / ones.sum().clamp_min(1e-8)
        self.assertAlmostEqual(float(weighted), float(per_sample.mean()), places=6)

    def test_partition_trainer_recency_flags_and_training_time_policy(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        defaults = module.parse_args([])
        self.assertEqual(defaults.recency_weighting, "none")
        self.assertEqual(defaults.recency_half_life_days, 120.0)
        self.assertEqual(defaults.recency_min_weight, 0.05)
        policy = module.build_training_time_policy(defaults, final_test_is_latest_available=True)
        self.assertEqual(policy["recency_weighting"], "none")
        self.assertTrue(policy["test_is_latest_period"])
        self.assertFalse(policy["test_used_for_recency_selection"])
        self.assertEqual(policy["checkpoint_selection"], "best_validation_return_then_fewer_order_legs")
        explicit = module.parse_args(
            ["--recency-weighting", "exponential", "--recency-half-life-days", "60", "--recency-min-weight", "0.1"]
        )
        explicit_policy = module.build_training_time_policy(explicit, final_test_is_latest_available=False)
        self.assertEqual(explicit_policy["recency_weighting"], "exponential")
        self.assertEqual(explicit_policy["recency_half_life_days"], 60.0)
        self.assertEqual(explicit_policy["recency_min_weight"], 0.1)
        self.assertFalse(explicit_policy["test_is_latest_period"])

    def test_recency_weighting_trains_and_is_uniform_when_disabled(self) -> None:
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
            RecencyWeightConfig,
            train_minute_to_hour_dqn,
        )

        def make_split(name: str, dates: list[str]) -> HourFromMinuteDataSplit:
            n = len(dates)
            return HourFromMinuteDataSplit(
                name=name,
                decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
                next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
                minute_feature_names=["m"],
                hour_feature_names=["h"],
                action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((n, 1, 1, 1)),
                minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((n, 1, 1)),
                action_returns=torch.zeros((n, 2)),
                action_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                # Last row has no successor -> valid_index_mask False so the env resets at the boundary.
                valid_start_indices=torch.arange(n - 1, dtype=torch.long),
                valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
                minute_feature_mean=torch.zeros(1),
                minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1),
                hour_feature_std=torch.ones(1),
                hours_lookback=1,
                minutes_per_hour=1,
            )

        train = make_split(
            "train", ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-02", "2026-05-20"]
        )
        val = make_split("val", ["2026-06-01", "2026-06-02"])
        learning = DQNLearningConfig(
            num_envs=2,
            episode_length=3,
            replay_capacity=64,
            batch_size=4,
            train_steps=8,
            warmup_steps=2,
            gamma=0.99,
            learning_rate=1e-3,
            weight_decay=0.0,
            target_update_interval=3,
            epsilon_start=0.2,
            epsilon_end=0.0,
            eval_interval=4,
            grad_clip=1.0,
            use_amp=False,
        )
        env = MinuteToHourEnvConfig(num_envs=2, episode_length=3)

        def run(mode: str) -> dict:
            config = MinuteToHourTrainingConfig(
                env=env,
                learning=learning,
                d_model=16,
                n_heads=2,
                minute_layers=1,
                hour_layers=1,
                feedforward_dim=16,
                action_embedding_dim=4,
                recency=RecencyWeightConfig(mode=mode, half_life_days=60.0, min_weight=0.05),
            )
            torch.manual_seed(0)
            _, artifacts = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config)
            return artifacts["recency_policy"]

        # Disabled (default): every training row keeps weight 1.0 -> weighted loss == plain mean.
        uniform = run("none")
        self.assertEqual(uniform["weight_min"], 1.0)
        self.assertEqual(uniform["weight_max"], 1.0)
        # Exponential: older training rows are down-weighted; weights stay within [min_weight, 1];
        # and the trainer never references the test split for recency.
        weighted = run("exponential")
        self.assertEqual(weighted["mode"], "exponential")
        self.assertFalse(weighted["test_used_for_recency_selection"])
        self.assertLess(weighted["weight_min"], weighted["weight_max"])
        self.assertGreaterEqual(weighted["weight_min"], 0.05)
        self.assertLessEqual(weighted["weight_max"], 1.0)

    def test_episode_truncation_is_not_terminal_but_data_boundary_is(self) -> None:
        module = __import__(
            "rl_quant.minute_to_hour_transformer",
            fromlist=["VectorizedMinuteToHourEnv", "MinuteToHourEnvConfig", "HourFromMinuteDataSplit"],
        )
        n = 4
        split = module.HourFromMinuteDataSplit(
            name="train",
            decision_timestamps=[f"2026-01-0{i + 1}T14:30:00+00:00" for i in range(n)],
            next_timestamps=[f"2026-01-0{i + 1}T15:30:00+00:00" for i in range(n)],
            minute_feature_names=["m"],
            hour_feature_names=["h"],
            action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n, 1, 1, 1)),
            minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1)),
            action_returns=torch.zeros((n, 2)),
            action_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            valid_start_indices=torch.arange(n - 1, dtype=torch.long),
            valid_index_mask=torch.tensor([True, True, True, False]),
            minute_feature_mean=torch.zeros(1),
            minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1),
            hour_feature_std=torch.ones(1),
            hours_lookback=1,
            minutes_per_hour=1,
        )
        env = module.VectorizedMinuteToHourEnv(
            split, module.MinuteToHourEnvConfig(num_envs=1, episode_length=2, initial_action=0), torch.device("cpu")
        )
        cash = torch.zeros(1, dtype=torch.long)
        # Reach the episode-length boundary with a valid next row -> truncation (done) but NOT terminal.
        env.reset(torch.ones(1, dtype=torch.bool))
        env.indices[:] = 0
        env.steps[:] = 0
        first = env.step(cash)  # steps 0->1, next row 1 is valid
        self.assertEqual(float(first["resets"][0].item()), 0.0)
        self.assertEqual(float(first["terminated"][0].item()), 0.0)
        second = env.step(cash)  # steps 1->2 == episode_length -> truncation
        self.assertEqual(float(second["resets"][0].item()), 1.0)
        self.assertEqual(float(second["terminated"][0].item()), 0.0)  # bootstrap must still happen
        # Stepping into a row whose successor is invalid -> a true data-boundary terminal.
        env.reset(torch.ones(1, dtype=torch.bool))
        env.indices[:] = 2
        env.steps[:] = 0
        boundary = env.step(cash)  # next row 3 has valid_index_mask False
        self.assertEqual(float(boundary["terminated"][0].item()), 1.0)
        self.assertEqual(float(boundary["resets"][0].item()), 1.0)

    def test_recency_weighting_rejects_zero_min_weight(self) -> None:
        from rl_quant.minute_to_hour_transformer import compute_recency_weights

        with self.assertRaises(ValueError):
            compute_recency_weights(
                ["2026-01-01T14:30:00+00:00"], 1, mode="exponential", half_life_days=60.0, min_weight=0.0
            )

    def test_recency_weighting_rejects_train_overlapping_validation(self) -> None:
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
            RecencyWeightConfig,
            train_minute_to_hour_dqn,
        )

        def make_split(name: str, dates: list[str]) -> HourFromMinuteDataSplit:
            n = len(dates)
            return HourFromMinuteDataSplit(
                name=name,
                decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
                next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
                minute_feature_names=["m"],
                hour_feature_names=["h"],
                action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((n, 1, 1, 1)),
                minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((n, 1, 1)),
                action_returns=torch.zeros((n, 2)),
                action_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                valid_start_indices=torch.arange(n - 1, dtype=torch.long),
                valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
                minute_feature_mean=torch.zeros(1),
                minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1),
                hour_feature_std=torch.ones(1),
                hours_lookback=1,
                minutes_per_hour=1,
            )

        # train_max (2026-06-15) is AFTER validation start (2026-06-01) -> recency must refuse.
        train = make_split("train", ["2026-05-01", "2026-06-15"])
        val = make_split("val", ["2026-06-01", "2026-06-02"])
        learning = DQNLearningConfig(
            num_envs=1,
            episode_length=2,
            replay_capacity=8,
            batch_size=2,
            train_steps=2,
            warmup_steps=1,
            gamma=0.99,
            learning_rate=1e-3,
            weight_decay=0.0,
            target_update_interval=2,
            epsilon_start=0.1,
            epsilon_end=0.0,
            eval_interval=2,
            grad_clip=1.0,
            use_amp=False,
        )
        config = MinuteToHourTrainingConfig(
            env=MinuteToHourEnvConfig(num_envs=1, episode_length=2),
            learning=learning,
            d_model=16,
            n_heads=2,
            minute_layers=1,
            hour_layers=1,
            feedforward_dim=16,
            action_embedding_dim=4,
            recency=RecencyWeightConfig(mode="exponential", half_life_days=60.0, min_weight=0.05),
        )
        with self.assertRaises(ValueError):
            train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config)

    def test_dqn_td_target_bootstraps_through_truncation_only_zeros_terminal(self) -> None:
        from rl_quant.core import dqn_td_target

        rewards = torch.tensor([2.0, 2.0])
        next_q = torch.tensor([10.0, 10.0])
        # First transition is a truncation (terminated=0 -> bootstrap); second is a true terminal.
        terminated = torch.tensor([0.0, 1.0])
        target = dqn_td_target(rewards, 0.9, terminated, next_q)
        self.assertAlmostEqual(float(target[0].item()), 2.0 + 0.9 * 10.0, places=5)
        self.assertAlmostEqual(float(target[1].item()), 2.0, places=5)

    def test_dqn_td_target_is_nan_safe_and_shape_checked(self) -> None:
        from rl_quant.core import dqn_td_target

        # A terminal row's NaN/Inf next_q must NOT propagate (torch.where selects 0 for terminals).
        rewards = torch.tensor([1.0, 1.0, 2.0])
        next_q = torch.tensor([float("nan"), float("inf"), 10.0])
        terminated = torch.tensor([1.0, 1.0, 0.0])
        target = dqn_td_target(rewards, 0.9, terminated, next_q)
        self.assertTrue(bool(torch.isfinite(target).all().item()))
        self.assertAlmostEqual(float(target[0].item()), 1.0, places=5)
        self.assertAlmostEqual(float(target[1].item()), 1.0, places=5)
        self.assertAlmostEqual(float(target[2].item()), 2.0 + 0.9 * 10.0, places=5)
        # A boolean terminated mask works identically.
        self.assertTrue(bool(torch.isfinite(dqn_td_target(rewards, 0.9, terminated.bool(), next_q)).all().item()))
        # Shape mismatch must raise (guards the silent (B, 1) vs (B,) broadcast bug).
        with self.assertRaises(ValueError):
            dqn_td_target(torch.zeros(3, 1), 0.9, torch.zeros(3), torch.zeros(3))

    def test_hourly_env_truncation_is_not_terminal_but_data_boundary_is(self) -> None:
        module = __import__(
            "rl_quant.hourly_transformer",
            fromlist=["VectorizedHourlyAllocationEnv", "HourlyEnvConfig", "HourlyDataSplit"],
        )
        n = 6
        split = module.HourlyDataSplit(
            name="train",
            timestamps=[f"2026-01-02T14:3{m}:00+00:00" for m in range(n)],
            next_timestamps=[f"2026-01-02T14:3{m + 1}:00+00:00" for m in range(n)],
            feature_names=["x"],
            action_names=["CASH", "QQQ"],
            features=torch.zeros((n, 1), dtype=torch.float32),
            action_returns=torch.zeros((n, 2), dtype=torch.float32),
            session_dates=["2026-01-02"] * n,
            valid_start_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            valid_index_mask=torch.tensor([True, True, True, True, True, False]),
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
            lookback=1,
            bar_interval="1m",
        )
        env = module.VectorizedHourlyAllocationEnv(
            split, module.HourlyEnvConfig(lookback=1, num_envs=1, episode_length=2, initial_action=0), torch.device("cpu")
        )
        cash = torch.zeros(1, dtype=torch.long)
        env.reset(torch.ones(1, dtype=torch.bool))
        env.indices[:] = 0
        env.steps[:] = 0
        first = env.step(cash)
        self.assertEqual(float(first["resets"][0].item()), 0.0)
        self.assertEqual(float(first["terminated"][0].item()), 0.0)
        second = env.step(cash)  # steps 1->2 == episode_length -> truncation, not terminal
        self.assertEqual(float(second["resets"][0].item()), 1.0)
        self.assertEqual(float(second["terminated"][0].item()), 0.0)
        env.reset(torch.ones(1, dtype=torch.bool))
        env.indices[:] = 4
        env.steps[:] = 0
        boundary = env.step(cash)  # next row 5 has no valid successor -> true terminal
        self.assertEqual(float(boundary["terminated"][0].item()), 1.0)
        self.assertEqual(float(boundary["resets"][0].item()), 1.0)

    def test_strict_latest_partition_rejects_duplicate_selected_labels(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        violations = module.strict_latest_partition_violations(
            selected_labels=["2026-01-03", "2026-01-03"],
            all_available_labels=["2026-01-01", "2026-01-02", "2026-01-03"],
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("duplicate" in violation for violation in violations))
        # Duplicates AND a non-latest final partition must both be reported in a single run
        # (no early-return masking, so the user fixes everything at once).
        combined = module.strict_latest_partition_violations(
            selected_labels=["2026-01-01", "2026-01-05", "2026-01-05"],
            all_available_labels=["2026-01-01", "2026-01-05", "2026-01-10"],
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("duplicate" in violation for violation in combined))
        self.assertTrue(any("not the latest available" in violation for violation in combined))
        # Duplicates must NOT suppress the missing-partition report: both surface in one run.
        dup_and_missing = module.strict_latest_partition_violations(
            selected_labels=["2026-01-01", "2026-01-03", "2026-01-03"],
            all_available_labels=["2026-01-01", "2026-01-02", "2026-01-03"],
            allow_truncated_training_history=False,
        )
        self.assertTrue(any("duplicate" in violation for violation in dup_and_missing))
        self.assertTrue(any("silently excluded" in violation for violation in dup_and_missing))

    def test_amp_precision_helpers_and_cuda_memory_report(self) -> None:
        from contextlib import nullcontext

        from rl_quant.core import (
            DQNLearningConfig,
            autocast_context,
            cuda_memory_report,
            make_grad_scaler,
            resolve_amp_dtype,
        )

        # Precision name -> dtype mapping, with eager rejection of unknown names.
        self.assertEqual(resolve_amp_dtype("fp16"), torch.float16)
        self.assertEqual(resolve_amp_dtype("bf16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            resolve_amp_dtype("fp8")
        cpu = torch.device("cpu")
        # On CPU, AMP is always disabled regardless of dtype -> a no-op nullcontext (no behavior change).
        self.assertIsInstance(autocast_context(cpu, True, "bf16"), type(nullcontext()))
        with autocast_context(cpu, True, "bf16"):
            pass
        # GradScaler is disabled off-CUDA; an invalid amp_dtype is rejected eagerly even when disabled.
        self.assertFalse(make_grad_scaler(cpu, True, "bf16").is_enabled())
        with self.assertRaises(ValueError):
            make_grad_scaler(cpu, False, "fp8")
        # cuda_memory_report is CPU-safe (zeros) so callers can log/guard unconditionally.
        report = cuda_memory_report(cpu)
        self.assertEqual(
            set(report), {"allocated_gb", "reserved_gb", "peak_allocated_gb", "peak_reserved_gb", "free_gb", "total_gb"}
        )
        self.assertEqual(report["total_gb"], 0.0)
        # Default precision is fp16 (backward compatible) and bf16 is accepted.
        self.assertEqual(
            DQNLearningConfig(
                num_envs=1, episode_length=1, replay_capacity=1, batch_size=1, train_steps=1, warmup_steps=0,
                gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=1, epsilon_start=0.0,
                epsilon_end=0.0, eval_interval=1, grad_clip=1.0,
            ).amp_dtype,
            "fp16",
        )

    def test_all_public_rl_quant_modules_import(self) -> None:
        import importlib
        import pkgutil

        import rl_quant

        failures = []
        for mod in pkgutil.iter_modules(rl_quant.__path__, "rl_quant."):
            try:
                importlib.import_module(mod.name)
            except Exception as exc:  # noqa: BLE001 - want to report every broken module
                failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
