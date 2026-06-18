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
from rl_quant.training.intraday import _apply_action_threshold  # noqa: E402
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

    def test_strict_latest_partition_uses_parsed_dates_not_lexicographic_order(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        fn = module.strict_latest_partition_violations
        # "Latest available" is by PARSED DATE, independent of input order: available passed UNSORTED
        # must still treat 2026-01-03 as latest, so a full, date-ordered selection is admissible.
        self.assertEqual(
            fn(
                selected_labels=["2026-01-01", "2026-01-02", "2026-01-03"],
                all_available_labels=["2026-01-03", "2026-01-01", "2026-01-02"],
                allow_truncated_training_history=False,
            ),
            [],
        )
        # The REAL partition label format is a date RANGE (start_to_end). chunk_dates produces
        # non-overlapping consecutive windows -> distinct start prefixes -> unambiguous; a full
        # date-ordered selection is admissible, and "latest" is order-independent (unsorted available).
        ranges = ["2026-01-01_to_2026-01-31", "2026-02-01_to_2026-02-28", "2026-03-01_to_2026-03-31"]
        self.assertEqual(
            fn(selected_labels=ranges, all_available_labels=ranges, allow_truncated_training_history=False), []
        )
        self.assertEqual(
            fn(
                selected_labels=ranges,
                all_available_labels=[ranges[2], ranges[0], ranges[1]],
                allow_truncated_training_history=False,
            ),
            [],
        )
        # Same-START-date distinct VALID range labels (a rebuild leaving two windows that share a
        # start) cannot be ordered from the start prefix -> fail closed with an ambiguity violation,
        # suppressing the later latest/coverage checks (meaningless without a defined order).
        ambiguous = fn(
            selected_labels=["2026-01-01_to_2026-01-15", "2026-01-01_to_2026-01-31"],
            all_available_labels=["2026-01-01_to_2026-01-15", "2026-01-01_to_2026-01-31"],
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("not chronologically unambiguous" in v for v in ambiguous))
        # Malformed labels are rejected as INVALID (the pattern is fully anchored, not a prefix match):
        # trailing garbage, a version suffix (2026-06-15_v2), and empty/inverted explicit ranges.
        for bad in (
            ["2026-06-15_v2", "2026-06-15_v10"],
            ["2026-01-01abc", "2026-01-02"],
            ["2026-01-01_to_2026-01-01"],
            ["2026-02-01_to_2026-01-01"],
        ):
            self.assertTrue(
                any(
                    "invalid labels" in v
                    for v in fn(selected_labels=bad, all_available_labels=bad, allow_truncated_training_history=True)
                ),
                bad,
            )
        # Distinct-start OVERLAPPING windows (a short window contained in a wide backfill) are rejected:
        # ranking by start would crown the contained 2026-03-15_to_2026-03-20 as latest even though the
        # container holds newer data (through Mar 31), and the windows leak train/test. Fail closed.
        overlapping = fn(
            selected_labels=["2026-01-01_to_2026-03-31", "2026-03-15_to_2026-03-20"],
            all_available_labels=["2026-01-01_to_2026-03-31", "2026-03-15_to_2026-03-20"],
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("overlap" in v for v in overlapping))
        # Adjacent consecutive windows that merely SHARE a boundary (end_i == start_{i+1}) are NOT
        # overlapping and remain admissible.
        adjacent = ["2026-01-02_to_2026-01-07", "2026-01-07_to_2026-01-11", "2026-01-11_to_2026-01-16"]
        self.assertEqual(
            fn(selected_labels=adjacent, all_available_labels=adjacent, allow_truncated_training_history=False), []
        )

    def test_chronological_latest_label_ranks_by_window_end(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        latest = module._chronological_latest_label
        # Unsorted bare dates -> max by date, not positional [-1].
        self.assertEqual(latest(["2026-01-03", "2026-01-01", "2026-01-02"]), "2026-01-03")
        # Ranges rank by END (most recent data): a wider window that ENDS later outranks a
        # later-STARTING but earlier-ENDING window.
        self.assertEqual(
            latest(["2026-02-01_to_2026-02-28", "2026-01-01_to_2026-03-31"]), "2026-01-01_to_2026-03-31"
        )
        # No parseable labels -> fall back to the last given; empty -> None.
        self.assertEqual(latest(["partition_9", "partition_10"]), "partition_10")
        self.assertIsNone(latest([]))

    def test_label_span_is_fully_anchored_and_rejects_malformed(self) -> None:
        from rl_quant.partition_protocol import label_span as span
        # Bare date == one-day half-open window [date, date + 1 day) (never empty).
        start, end = span("2026-01-01")
        self.assertEqual((end - start).days, 1)
        # Valid explicit range, end exclusive.
        start, end = span("2026-01-01_to_2026-02-01")
        self.assertEqual((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")), ("2026-01-01", "2026-02-01"))
        # Fully anchored: trailing garbage / version suffix is rejected, NOT silently truncated.
        for bad in ("2026-01-01abc", "2026-01-01_to_2026-02-01_garbage", "2026-06-15_v2"):
            self.assertIsNone(span(bad), bad)
        # Empty, inverted, and impossible-date ranges are malformed.
        for bad in ("2026-01-01_to_2026-01-01", "2026-02-01_to_2026-01-01", "2026-99-99", "2026-01-01_to_2026-99-99"):
            self.assertIsNone(span(bad), bad)

    def test_strict_latest_validates_selected_labels_in_all_modes(self) -> None:
        module = load_script("train_hourly_from_second_protocol_partitions")
        fn = module.strict_latest_partition_violations
        available = ["2026-01-01_to_2026-02-01", "2026-02-01_to_2026-03-01"]
        # An UNKNOWN selected label (well-formed but not on disk) is reported even when truncated
        # history is allowed -- it is never admissible, independent of the coverage override.
        unknown = fn(
            selected_labels=["2026-01-01_to_2026-02-01", "2026-06-01_to_2026-07-01"],
            all_available_labels=available,
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("unknown labels" in v for v in unknown))
        # An INVALID selected label is validated too (the parser scans selected, not just available).
        invalid_selected = fn(
            selected_labels=["2026-13-99_to_2026-99-01"],
            all_available_labels=available,
            allow_truncated_training_history=True,
        )
        self.assertTrue(any("invalid labels" in v for v in invalid_selected))
        # The overlap diagnostic names BOTH the overlapping window and the container it overlaps.
        overlap = fn(
            selected_labels=["2026-01-01_to_2026-03-31", "2026-03-15_to_2026-03-20"],
            all_available_labels=["2026-01-01_to_2026-03-31", "2026-03-15_to_2026-03-20"],
            allow_truncated_training_history=True,
        )
        joined = " ".join(overlap)
        self.assertIn("2026-03-15_to_2026-03-20", joined)
        self.assertIn("2026-01-01_to_2026-03-31", joined)

    def test_partition_protocol_shared_by_both_trainers(self) -> None:
        # Both training scripts must use the SAME latest-period gate implementation, so a fix in one
        # path cannot leave the other behind (the calendar-holdout drift this shared module closes).
        from rl_quant import partition_protocol

        protocol = load_script("train_hourly_from_second_protocol_partitions")
        self.assertIs(
            protocol.strict_latest_partition_violations, partition_protocol.strict_latest_partition_violations
        )
        script_path = ROOT / "scripts" / "train_hourly_from_second_calendar_holdout.py"
        spec = importlib.util.spec_from_file_location("calendar_holdout_shared_check", script_path)
        calendar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calendar)
        self.assertIs(
            calendar.strict_latest_partition_violations, partition_protocol.strict_latest_partition_violations
        )
        self.assertIs(calendar.label_span, partition_protocol.label_span)

    def test_calendar_holdout_selection_reportability(self) -> None:
        script_path = ROOT / "scripts" / "train_hourly_from_second_calendar_holdout.py"
        spec = importlib.util.spec_from_file_location("calendar_holdout_reportability", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("2026-01-01_to_2026-02-01", "2026-02-01_to_2026-03-01", "2026-03-01_to_2026-04-01"):
                (root / label).mkdir(parents=True)
                (root / label / "hour_from_second_dataset.pt").touch()
            # Clean run over ALL partitions, ending at the latest available -> reportable, not manual.
            args = module.parse_args(["--partitions-root", str(root), "--test-months", "1", "--val-months", "1"])
            errors, manual = module.calendar_selection_reportability(args, module.partition_paths(args))
            self.assertEqual(errors, [])
            self.assertFalse(manual)
            # Restricting to a non-latest suffix (--end-partition before the latest available) -> manual
            # AND non-reportable, with the latest-availability violation named ("latest among selected"
            # is NOT enough when newer complete partitions exist).
            args2 = module.parse_args(
                [
                    "--partitions-root", str(root),
                    "--end-partition", "2026-02-01_to_2026-03-01",
                    "--test-months", "1", "--val-months", "1",
                ]
            )
            errors2, manual2 = module.calendar_selection_reportability(args2, module.partition_paths(args2))
            self.assertTrue(manual2)
            self.assertTrue(any("not the latest available" in e for e in errors2), errors2)

    def test_derive_reportable_partition_split_latest_suffix(self) -> None:
        from rl_quant.partition_protocol import derive_reportable_partition_split, partition_windows_from_labels

        labels = [f"2026-{m:02d}-01_to_2026-{m + 1:02d}-01" for m in range(1, 7)]  # 6 consecutive months
        windows = partition_windows_from_labels(labels)
        # val=2, test=1 -> test=[P5], val=[P3,P4], train=[P0,P1,P2].
        split = derive_reportable_partition_split(windows, val_count=2, test_count=1)
        self.assertEqual([w.label for w in split.test], [labels[5]])
        self.assertEqual([w.label for w in split.val], [labels[3], labels[4]])
        self.assertEqual([w.label for w in split.train], labels[:3])
        # val=2, test=3 -> test=last 3, val=preceding 2, train=first 1.
        split2 = derive_reportable_partition_split(windows, val_count=2, test_count=3)
        self.assertEqual([w.label for w in split2.test], labels[3:])
        self.assertEqual([w.label for w in split2.val], [labels[1], labels[2]])
        self.assertEqual([w.label for w in split2.train], [labels[0]])
        # Latest is by window END, independent of input order.
        shuffled = partition_windows_from_labels([labels[2], labels[5], labels[0], labels[4], labels[1], labels[3]])
        latest_test = derive_reportable_partition_split(shuffled, val_count=1, test_count=1).test
        self.assertEqual([w.label for w in latest_test], [labels[5]])

    def test_derive_reportable_partition_split_guards(self) -> None:
        from datetime import datetime as _datetime

        from rl_quant.partition_protocol import (
            PartitionWindow,
            derive_reportable_partition_split,
            partition_windows_from_labels,
        )

        labels = [f"2026-{m:02d}-01_to_2026-{m + 1:02d}-01" for m in range(1, 5)]  # 4 months
        windows = partition_windows_from_labels(labels)
        with self.assertRaises(ValueError):  # need >= val+test+1 = 5, have 4
            derive_reportable_partition_split(windows, val_count=2, test_count=2)
        with self.assertRaises(ValueError):  # non-positive counts
            derive_reportable_partition_split(windows, val_count=0, test_count=1)
        with self.assertRaises(ValueError):  # truncation must be explicitly allowed
            derive_reportable_partition_split(windows, val_count=1, test_count=1, train_window_count=1)
        truncated = derive_reportable_partition_split(
            windows, val_count=1, test_count=1, train_window_count=1, allow_truncated_training_history=True
        )
        self.assertEqual([w.label for w in truncated.train], [labels[1]])  # most RECENT train block kept
        # Overlapping windows are not a valid walk-forward.
        overlapping = [
            PartitionWindow("a", _datetime(2026, 1, 1), _datetime(2026, 3, 1)),
            PartitionWindow("b", _datetime(2026, 2, 1), _datetime(2026, 4, 1)),
            PartitionWindow("c", _datetime(2026, 4, 1), _datetime(2026, 5, 1)),
        ]
        with self.assertRaises(ValueError):
            derive_reportable_partition_split(overlapping, val_count=1, test_count=1)
        # Incomplete windows are dropped, so test is the latest COMPLETE window.
        incomplete = [*windows[:3], PartitionWindow(windows[3].label, windows[3].start, windows[3].end_exclusive, complete=False)]
        self.assertEqual(
            [w.label for w in derive_reportable_partition_split(incomplete, val_count=1, test_count=1).test],
            [labels[2]],
        )
        # Malformed labels are rejected at parse time, not silently dropped.
        with self.assertRaises(ValueError):
            partition_windows_from_labels(["2026-01-01_to_garbage"])

    def test_build_split_records_and_gates_latest_reward_row_filtering(self) -> None:
        module = __import__("rl_quant.minute_to_hour_transformer", fromlist=["_build_split"])

        def _payload(invalid_row: int) -> dict:
            returns = torch.tensor([[0.0, 0.01], [0.0, 0.02], [0.0, 0.03]], dtype=torch.float32)
            label_valid = torch.ones((3, 2), dtype=torch.bool)
            label_valid[invalid_row, 1] = False
            returns[invalid_row, 1] = float("nan")  # contract: a label-invalid return must be NaN
            return {
                "decision_timestamps": [f"2026-06-10T1{5 + i}:30:00+00:00" for i in range(3)],
                "next_timestamps": [f"2026-06-10T1{6 + i}:30:00+00:00" for i in range(3)],
                "minute_timestamp_grid": [[[f"2026-06-10T1{5 + i}:29:59+00:00"]] for i in range(3)],
                "minute_feature_names": ["m"],
                "hour_feature_names": ["h"],
                "action_names": ["CASH", "QQQ"],
                "minute_features": torch.zeros((3, 1, 1, 1), dtype=torch.float32),
                "minute_mask": torch.ones((3, 1, 1), dtype=torch.bool),
                "hour_features": torch.zeros((3, 1, 1), dtype=torch.float32),
                "action_returns": returns,
                "action_valid_mask": torch.ones((3, 2), dtype=torch.bool),
                "label_valid_mask": label_valid,
                "source_bar_interval": "1s",
                "context_bars_per_hour": 3600,
                "minutes_per_hour": 3600,
                "decision_grid_minutes": 60,
                "bar_latency_ms": 1000,
            }

        # The LATEST row (index 2) has a selectable non-cash missing label -> the filter drops it, so the
        # TEST split loses its latest reward row. Recorded AND gated non-reportable (not silently shrunk).
        test_split = module._build_split(name="test", payload=_payload(2))
        self.assertEqual(test_split.excluded_missing_label_rows, 1)
        self.assertTrue(test_split.filter_removed_latest_reward_rows)
        self.assertFalse(test_split.dataset_reportable)
        self.assertIn("test_filter_removed_latest_reward_rows", test_split.dataset_reportability_errors)
        self.assertEqual(test_split.valid_start_indices.tolist(), [0, 1])
        # Same data as a TRAIN split: the drop is recorded but NOT gated by this rule (train may
        # legitimately end before the latest reward row).
        train_split = module._build_split(name="train", payload=_payload(2))
        self.assertTrue(train_split.filter_removed_latest_reward_rows)
        self.assertNotIn("test_filter_removed_latest_reward_rows", train_split.dataset_reportability_errors)
        # Dropping a NON-latest row (index 0) does not trip the latest-reward gate.
        mid_split = module._build_split(name="test", payload=_payload(0))
        self.assertEqual(mid_split.excluded_missing_label_rows, 1)
        self.assertFalse(mid_split.filter_removed_latest_reward_rows)
        self.assertNotIn("test_filter_removed_latest_reward_rows", mid_split.dataset_reportability_errors)

    def test_transition_feature_table_encodes_hold_switch_exit(self) -> None:
        from rl_quant.trading_constraints import TRANSITION_FEATURE_NAMES, build_transition_feature_table

        # 0=CASH, 1=QQQ (lev 1, group 1), 2=SQQQ (lev 1, group 1).
        table = build_transition_feature_table(
            action_count=3,
            cash_index=0,
            one_way_cost_bps=2.0,
            extra_switch_penalty_bps=1.0,
            count_etf_to_etf_as_two_legs=True,
            action_leverage=torch.tensor([0.0, 1.0, 1.0]),
            action_group_ids=torch.tensor([0, 1, 1]),
            device="cpu",
        )
        self.assertEqual(tuple(table.shape), (3, 3, len(TRANSITION_FEATURE_NAMES)))
        col = {name: i for i, name in enumerate(TRANSITION_FEATURE_NAMES)}
        # hold (cash->cash): 0 legs, is_hold=1, is_switch=0.
        self.assertEqual(table[0, 0, col["legs"]].item(), 0.0)
        self.assertEqual(table[0, 0, col["is_hold"]].item(), 1.0)
        self.assertEqual(table[0, 0, col["is_switch"]].item(), 0.0)
        # cash->etf: 1 leg, cost = 1*2 + 1 switch penalty = 3 bps; prev_is_cash=1, cand_is_cash=0.
        self.assertEqual(table[0, 1, col["legs"]].item(), 1.0)
        self.assertAlmostEqual(table[0, 1, col["est_cost_bps_over_100"]].item(), 3.0 / 100.0, places=6)
        self.assertEqual(table[0, 1, col["prev_is_cash"]].item(), 1.0)
        self.assertEqual(table[0, 1, col["cand_is_cash"]].item(), 0.0)
        self.assertAlmostEqual(table[0, 1, col["leverage_delta"]].item(), 1.0, places=6)
        self.assertEqual(table[0, 1, col["same_group"]].item(), 0.0)  # cash group != QQQ group
        # etf->etf switch (QQQ->SQQQ): 2 legs, cost = 2*2 + 1 = 5 bps, same group.
        self.assertEqual(table[1, 2, col["legs"]].item(), 2.0)
        self.assertAlmostEqual(table[1, 2, col["est_cost_bps_over_100"]].item(), 5.0 / 100.0, places=6)
        self.assertEqual(table[1, 2, col["same_group"]].item(), 1.0)

    def test_build_dynamic_transition_features(self) -> None:
        # PR-D D1: additive per-env dynamic position-state features (P&L excursion). Static path untouched
        # -> training byte-identical (covered by the unchanged trainer tests). Nothing consumes this until D3.
        from rl_quant.trading_constraints import (
            DYNAMIC_TRANSITION_FEATURE_DIM,
            DYNAMIC_TRANSITION_FEATURE_NAMES,
            build_dynamic_transition_features,
        )

        self.assertEqual(DYNAMIC_TRANSITION_FEATURE_DIM, len(DYNAMIC_TRANSITION_FEATURE_NAMES))
        self.assertEqual(DYNAMIC_TRANSITION_FEATURE_NAMES[0], "unrealized_pnl")

        upnl = torch.tensor([0.10, -0.20, 2.0])
        mae = torch.tensor([-0.05, -0.30, -0.10])
        mfe = torch.tensor([0.20, 0.00, 0.50])
        out = build_dynamic_transition_features(unrealized_pnl=upnl, mae=mae, mfe=mfe, clamp=1.0)
        self.assertEqual(tuple(out.shape), (3, DYNAMIC_TRANSITION_FEATURE_DIM))
        col = {name: i for i, name in enumerate(DYNAMIC_TRANSITION_FEATURE_NAMES)}
        # row 0: drawdown_from_peak = mfe - upnl = 0.10; runup_from_trough = upnl - mae = 0.15.
        self.assertAlmostEqual(out[0, col["drawdown_from_peak"]].item(), 0.10, places=6)
        self.assertAlmostEqual(out[0, col["runup_from_trough"]].item(), 0.15, places=6)
        # row 2: upnl 2.0 clamps to 1.0; drawdown = mfe(0.5) - upnl(1.0) = -0.5 -> clamped to >= 0.
        self.assertAlmostEqual(out[2, col["unrealized_pnl"]].item(), 1.0, places=6)
        self.assertAlmostEqual(out[2, col["drawdown_from_peak"]].item(), 0.0, places=6)
        self.assertAlmostEqual(out[2, col["runup_from_trough"]].item(), 1.0 - (-0.10), places=6)
        # Derived spreads are always non-negative; deterministic.
        self.assertTrue(bool((out[:, col["drawdown_from_peak"]] >= 0).all().item()))
        self.assertTrue(bool((out[:, col["runup_from_trough"]] >= 0).all().item()))
        self.assertTrue(torch.equal(out, build_dynamic_transition_features(unrealized_pnl=upnl, mae=mae, mfe=mfe, clamp=1.0)))

    def test_qnetwork_transition_features_condition_q_on_held_position(self) -> None:
        from rl_quant.minute_to_hour_transformer import MinuteToHourCausalTransformerQNetwork
        from rl_quant.trading_constraints import TRANSITION_FEATURE_DIM

        action_count, d_model, batch = 3, 16, 2
        ctor = dict(
            minute_feature_dim=1, hour_feature_dim=1, action_count=action_count, hours_lookback=1,
            minutes_per_hour=1, d_model=d_model, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16,
        )
        inputs = dict(
            minute_features=torch.zeros(batch, 1, 1, 1),
            minute_mask=torch.ones(batch, 1, 1, dtype=torch.bool),
            hour_features=torch.zeros(batch, 1, 1),
            constraint_features=torch.zeros(batch, 6),
        )
        prev0 = torch.zeros(batch, dtype=torch.long)
        prev1 = torch.ones(batch, dtype=torch.long)

        def _contribution(net, *, prev, action_features):
            net.eval()
            with torch.no_grad():
                full = net(previous_actions=prev, action_features=action_features, **inputs)
                net.transition_encoder, net.transition_bias = None, None  # isolate the transition path
                base = net(previous_actions=prev, action_features=action_features, **inputs)
            return full - base

        table = torch.randn(action_count, action_count, TRANSITION_FEATURE_DIM)
        af = torch.zeros(batch, action_count, 2)
        # Action-conditioned branch: zero-init -> the transition path contributes nothing at init.
        torch.manual_seed(1)
        net = MinuteToHourCausalTransformerQNetwork(
            action_feature_dim=2, transition_feature_dim=TRANSITION_FEATURE_DIM, transition_table=table, **ctor
        )
        self.assertTrue(
            torch.allclose(_contribution(net, prev=prev0, action_features=af), torch.zeros(batch, action_count))
        )

        def _perturbed(seed):
            torch.manual_seed(seed)
            model = MinuteToHourCausalTransformerQNetwork(
                action_feature_dim=2, transition_feature_dim=TRANSITION_FEATURE_DIM, transition_table=table, **ctor
            )
            with torch.no_grad():  # varied (not constant) weights so LayerNorm preserves the signal
                model.transition_encoder[0].weight.copy_(torch.arange(d_model * TRANSITION_FEATURE_DIM).float().reshape(d_model, TRANSITION_FEATURE_DIM) * 0.01)
                model.transition_encoder[0].bias.copy_(torch.arange(d_model).float() * 0.01)
            return model

        contrib0 = _contribution(_perturbed(1), prev=prev0, action_features=af)
        contrib1 = _contribution(_perturbed(1), prev=prev1, action_features=af)
        self.assertFalse(torch.allclose(contrib0, torch.zeros_like(contrib0)))  # now contributes
        self.assertFalse(torch.allclose(contrib0, contrib1))  # contribution depends on held position
        # Fallback head (no action_features): forward returns [B, A] and is held-position aware via the bias.
        torch.manual_seed(2)
        fb = MinuteToHourCausalTransformerQNetwork(
            action_feature_dim=0, transition_feature_dim=TRANSITION_FEATURE_DIM, transition_table=table, **ctor
        )
        with torch.no_grad():
            out = fb(previous_actions=prev0, action_features=None, **inputs)
        self.assertEqual(tuple(out.shape), (batch, action_count))

        def _perturbed_fallback(seed):
            torch.manual_seed(seed)
            model = MinuteToHourCausalTransformerQNetwork(
                action_feature_dim=0, transition_feature_dim=TRANSITION_FEATURE_DIM, transition_table=table, **ctor
            )
            with torch.no_grad():
                model.transition_bias.weight.copy_(torch.arange(TRANSITION_FEATURE_DIM).float().reshape(1, -1) * 0.1)
                model.transition_bias.bias.fill_(0.0)
            return model

        self.assertFalse(
            torch.allclose(
                _contribution(_perturbed_fallback(2), prev=prev0, action_features=None),
                _contribution(_perturbed_fallback(2), prev=prev1, action_features=None),
            )
        )

    def test_qnetwork_dynamic_features_zero_init_and_condition_q(self) -> None:
        # PR-D D3a: the network can consume a per-env dynamic position-state vector. Zero-init -> a freshly
        # built dynamic-aware net scores IDENTICALLY (so flag-on-but-untrained == off, byte-identical); after
        # perturbing the encoder the Q depends on the dynamic state; dynamic_feature_dim=0 adds no params.
        from rl_quant.minute_to_hour_transformer import MinuteToHourCausalTransformerQNetwork
        from rl_quant.trading_constraints import DYNAMIC_TRANSITION_FEATURE_DIM

        action_count, d_model, batch = 3, 16, 2
        D = DYNAMIC_TRANSITION_FEATURE_DIM
        ctor = dict(
            minute_feature_dim=1, hour_feature_dim=1, action_count=action_count, hours_lookback=1,
            minutes_per_hour=1, d_model=d_model, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16,
        )
        inputs = dict(
            minute_features=torch.zeros(batch, 1, 1, 1),
            minute_mask=torch.ones(batch, 1, 1, dtype=torch.bool),
            hour_features=torch.zeros(batch, 1, 1),
            constraint_features=torch.zeros(batch, 6),
            previous_actions=torch.zeros(batch, dtype=torch.long),
        )
        af = torch.zeros(batch, action_count, 2)
        dyn = torch.randn(batch, D)

        # dynamic_feature_dim=0 -> no dynamic params at all.
        torch.manual_seed(1)
        off = MinuteToHourCausalTransformerQNetwork(action_feature_dim=2, **ctor)
        self.assertIsNone(off.dynamic_encoder)
        self.assertIsNone(off.dynamic_bias)
        self.assertFalse(any("dynamic" in k for k in off.state_dict()))

        # Zero-init: passing dynamic_state changes nothing at init (action-feature head + fallback head).
        for afd, action_features in ((2, af), (0, None)):
            torch.manual_seed(3)
            net = MinuteToHourCausalTransformerQNetwork(action_feature_dim=afd, dynamic_feature_dim=D, **ctor)
            net.eval()
            with torch.no_grad():
                base = net(action_features=action_features, dynamic_state=torch.zeros(batch, D), **inputs)
                withdyn = net(action_features=action_features, dynamic_state=dyn, **inputs)
            self.assertTrue(torch.allclose(base, withdyn), f"zero-init must be a no-op (afd={afd})")

        # After perturbing the dynamic encoder, the Q depends on the dynamic state (and differs across states).
        torch.manual_seed(3)
        net = MinuteToHourCausalTransformerQNetwork(action_feature_dim=2, dynamic_feature_dim=D, **ctor)
        with torch.no_grad():
            net.dynamic_encoder[0].weight.copy_(torch.arange(d_model * D).float().reshape(d_model, D) * 0.01)
            net.dynamic_encoder[0].bias.copy_(torch.arange(d_model).float() * 0.01)
        net.eval()
        with torch.no_grad():
            q_zero = net(action_features=af, dynamic_state=torch.zeros(batch, D), **inputs)
            q_a = net(action_features=af, dynamic_state=torch.zeros(batch, D) + 0.5, **inputs)
            q_b = net(action_features=af, dynamic_state=torch.zeros(batch, D) - 0.5, **inputs)
        self.assertFalse(torch.allclose(q_a, q_zero))  # dynamic state now moves Q vs the zero ablation
        self.assertFalse(torch.allclose(q_a, q_b))  # and different states give different Q
        # A dynamic-built model must fail closed when dynamic_state is omitted -- no silent non-dynamic scoring.
        with self.assertRaises(ValueError):
            net(action_features=af, dynamic_state=None, **inputs)

    def test_transition_features_clean_perturbation(self) -> None:
        # Reorg review: enabling transition_feature_dim>0 must be a CLEAN perturbation -- the shared backbone
        # (hour_encoder + head) is bit-identical to the transition_feature_dim=0 model under the same seed,
        # because the zero-init transition module is built under saved/restored RNG. So a transition A/B
        # isolates the feature, not a different random initialisation. Covers the action-feature + fallback heads.
        from rl_quant.minute_to_hour_transformer import MinuteToHourCausalTransformerQNetwork
        from rl_quant.trading_constraints import TRANSITION_FEATURE_DIM

        action_count, d_model, batch = 3, 16, 2
        f_dim = TRANSITION_FEATURE_DIM
        ctor = dict(
            minute_feature_dim=1, hour_feature_dim=1, action_count=action_count, hours_lookback=1,
            minutes_per_hour=1, d_model=d_model, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16,
        )
        inputs = dict(
            minute_features=torch.zeros(batch, 1, 1, 1),
            minute_mask=torch.ones(batch, 1, 1, dtype=torch.bool),
            hour_features=torch.zeros(batch, 1, 1),
            constraint_features=torch.zeros(batch, 6),
            previous_actions=torch.zeros(batch, dtype=torch.long),
        )
        zero_table = torch.zeros(action_count, action_count, f_dim)
        for afd, action_features in ((2, torch.zeros(batch, action_count, 2)), (0, None)):
            torch.manual_seed(7)
            off = MinuteToHourCausalTransformerQNetwork(action_feature_dim=afd, **ctor)
            torch.manual_seed(7)
            on = MinuteToHourCausalTransformerQNetwork(
                action_feature_dim=afd, transition_feature_dim=f_dim, transition_table=zero_table, **ctor
            )
            for sub in ("hour_encoder", "head"):  # backbone modules built AFTER the transition block
                for (name, p_off), (_, p_on) in zip(
                    getattr(off, sub).named_parameters(), getattr(on, sub).named_parameters()
                ):
                    self.assertTrue(torch.equal(p_off, p_on), f"{sub}.{name} differs (afd={afd}) -> RNG perturbed")
            off.eval()
            on.eval()
            with torch.no_grad():
                q_off = off(action_features=action_features, **inputs)
                q_on = on(action_features=action_features, **inputs)
            self.assertTrue(torch.equal(q_off, q_on), f"zero-table transition must be a no-op at init (afd={afd})")

    def test_td_next_q_max_depends_on_next_previous_action(self) -> None:
        # The TD target uses max_a Q(s', a) evaluated with next_previous_actions (the post-action held
        # position). This is a direct regression guard that a refactor passing the wrong previous action
        # to the target network would be caught: same next-market, different next position -> different max-Q.
        from rl_quant.minute_to_hour_transformer import MinuteToHourCausalTransformerQNetwork
        from rl_quant.trading_constraints import TRANSITION_FEATURE_DIM

        action_count, d_model, batch = 3, 16, 2
        torch.manual_seed(3)
        table = torch.randn(action_count, action_count, TRANSITION_FEATURE_DIM)
        net = MinuteToHourCausalTransformerQNetwork(
            minute_feature_dim=1, hour_feature_dim=1, action_count=action_count, hours_lookback=1,
            minutes_per_hour=1, d_model=d_model, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16,
            action_feature_dim=2, transition_feature_dim=TRANSITION_FEATURE_DIM, transition_table=table,
        )
        net.eval()
        with torch.no_grad():  # perturb so the (zero-init) transition path is active
            net.transition_encoder[0].weight.copy_(
                torch.arange(d_model * TRANSITION_FEATURE_DIM).float().reshape(d_model, TRANSITION_FEATURE_DIM) * 0.01
            )
            net.transition_encoder[0].bias.copy_(torch.arange(d_model).float() * 0.01)
        inputs = dict(
            minute_features=torch.zeros(batch, 1, 1, 1), minute_mask=torch.ones(batch, 1, 1, dtype=torch.bool),
            hour_features=torch.zeros(batch, 1, 1), constraint_features=torch.zeros(batch, 6),
            action_features=torch.zeros(batch, action_count, 2),
        )
        with torch.no_grad():
            max_from_cash = net(previous_actions=torch.zeros(batch, dtype=torch.long), **inputs).max(dim=1).values
            max_from_qqq = net(previous_actions=torch.ones(batch, dtype=torch.long), **inputs).max(dim=1).values
        self.assertFalse(torch.allclose(max_from_cash, max_from_qqq))
        # The table gather guards out-of-range ids (both bounds) with a clear error rather than a silent
        # negative-index wrap to the last row.
        with self.assertRaises(ValueError):
            net._transition_rows(torch.tensor([-1], dtype=torch.long))
        with self.assertRaises(ValueError):
            net._transition_rows(torch.tensor([action_count], dtype=torch.long))

    def test_transition_table_cost_matches_env_reward_convention(self) -> None:
        from rl_quant.trading_constraints import TRANSITION_FEATURE_NAMES, build_transition_feature_table, trade_legs

        one_way, extra = 1.5, 0.5
        table = build_transition_feature_table(
            action_count=3, cash_index=0, one_way_cost_bps=one_way, extra_switch_penalty_bps=extra,
            count_etf_to_etf_as_two_legs=True, action_leverage=torch.tensor([0.0, 1.0, 1.0]),
            action_group_ids=torch.tensor([0, 1, 1]), device="cpu",
        )
        col = TRANSITION_FEATURE_NAMES.index("est_cost_bps_over_100")
        for prev in range(3):
            for cand in range(3):
                legs = trade_legs(
                    torch.tensor(prev), torch.tensor(cand), cash_index=0, count_etf_to_etf_as_two_legs=True
                ).item()
                # Must equal the env reward's cost deduction convention: legs*one_way + switch*extra.
                expected_bps = legs * one_way + (1.0 if prev != cand else 0.0) * extra
                self.assertAlmostEqual(table[prev, cand, col].item() * 100.0, expected_bps, places=5)

    def test_warm_start_schema_rejects_transition_mismatch(self) -> None:
        from rl_quant.minute_to_hour_transformer import _assert_checkpoint_schema
        from rl_quant.trading_constraints import CONSTRAINT_FEATURE_NAMES, TRANSITION_FEATURE_NAMES

        common = dict(
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"], action_feature_names=[]
        )
        base = {
            "minute_feature_names": ["m"], "hour_feature_names": ["h"], "action_names": ["CASH", "QQQ"],
            "action_feature_names": [], "constraint_feature_names": list(CONSTRAINT_FEATURE_NAMES),
        }
        names = list(TRANSITION_FEATURE_NAMES)
        # off->off and on->on match; on->off and off->on are rejected with a clear schema error.
        _assert_checkpoint_schema({**base, "transition_feature_names": []}, **common, transition_feature_dim=0)
        _assert_checkpoint_schema({**base, "transition_feature_names": names}, **common, transition_feature_dim=len(names))
        with self.assertRaises(ValueError):
            _assert_checkpoint_schema({**base, "transition_feature_names": names}, **common, transition_feature_dim=0)
        with self.assertRaises(ValueError):
            _assert_checkpoint_schema({**base, "transition_feature_names": []}, **common, transition_feature_dim=len(names))

    def test_execution_simulator_transition_cases(self) -> None:
        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            MarketSnapshot,
            PositionState,
            TerminalPolicy,
            fill_index,
            simulate_transition,
        )

        cfg = ExecutionConfig(  # trade_scale = 2*100 = 200; delayed_close (mid proxy)
            trade_lot_size=2, commission_per_share=0.01, extra_cost_per_share=0.02, terminal_policy=TerminalPolicy.CARRY
        )
        now = MarketSnapshot(mid=100.0, half_spread=0.05)
        fill = MarketSnapshot(mid=101.0, half_spread=0.05)
        nxt = MarketSnapshot(mid=103.0, half_spread=0.05)

        def run(old, new, *, terminal=False, config=cfg, n=now, f=fill, x=nxt):
            return simulate_transition(PositionState(position=old), new, n, f, x, is_terminal=terminal, config=config)

        # fill_index: min(now+latency, next), capped at next; latency<=0 collapses to current bar.
        self.assertEqual(fill_index(10, step_horizon=5, latency_steps=2), 12)
        self.assertEqual(fill_index(10, step_horizon=5, latency_steps=0), 10)
        self.assertEqual(fill_index(10, step_horizon=5, latency_steps=99), 15)
        # The vectorized fill_indices now lives in execution.py (single source of truth) and the intraday
        # env/pretraining sites import it; it must equal scalar fill_index applied element-wise. Lock it
        # against drift across negative/zero/positive latency, and confirm device/dtype are preserved.
        from rl_quant.execution import fill_indices as exec_fill_indices

        for horizon in (1, 5):
            for latency in (-3, 0, 1, 3, 99):
                idx = torch.arange(20)
                vec = exec_fill_indices(idx, step_horizon=horizon, latency_steps=latency)
                want = [fill_index(int(i), step_horizon=horizon, latency_steps=latency) for i in idx.tolist()]
                self.assertEqual(vec.tolist(), want)
                self.assertEqual(vec.dtype, idx.dtype)
                self.assertEqual(vec.device, idx.device)
        # Vectorized helper carries the same step_horizon>0 guard as the scalar version.
        with self.assertRaises(ValueError):
            exec_fill_indices(torch.arange(4), step_horizon=0, latency_steps=1)
        # Also assert the intraday env/pretraining sites import THIS helper (no private duplicate remains).
        import rl_quant.intraday_dqn as _idqn

        self.assertFalse(hasattr(_idqn, "_fill_indices"))
        self.assertIs(_idqn.compute_fill_indices, exec_fill_indices)

        per_share = 0.05 + 0.02 + 0.01  # half_spread + extra + commission
        # cash->cash: everything zero.
        z = run(0.0, 0.0)
        self.assertEqual((z.gross_return, z.entry_cost, z.exit_cost, z.net_return, z.order_legs), (0.0, 0.0, 0.0, 0.0, 0.0))
        # cash->asset (+1): no old leg, 1-unit entry cost, new earns fill->next.
        ca = run(0.0, 1.0)
        self.assertAlmostEqual(ca.old_latency_return, 0.0)
        self.assertAlmostEqual(ca.new_interval_return, 1.0 * (103.0 - 101.0) * 200.0)
        self.assertAlmostEqual(ca.entry_cost, 1.0 * per_share * 200.0)
        self.assertEqual(ca.order_legs, 1.0)
        # asset->same (+1->+1): NO re-entry cost, one continuous leg now->next.
        hold = run(1.0, 1.0)
        self.assertEqual(hold.entry_cost, 0.0)
        self.assertEqual(hold.order_legs, 0.0)
        self.assertAlmostEqual(hold.net_return, 1.0 * (103.0 - 100.0) * 200.0)
        # asset->cash (+1->0): old STILL earns the now->fill latency leg; 1-unit exit turnover cost.
        ac = run(1.0, 0.0)
        self.assertAlmostEqual(ac.old_latency_return, 1.0 * (101.0 - 100.0) * 200.0)
        self.assertEqual(ac.new_interval_return, 0.0)
        self.assertAlmostEqual(ac.entry_cost, 1.0 * per_share * 200.0)
        # A->B full reversal (-1 -> +1): turnover is 2 units.
        ab = run(-1.0, 1.0)
        self.assertEqual(ab.order_legs, 2.0)
        self.assertAlmostEqual(ab.entry_cost, 2.0 * per_share * 200.0)
        # Terminal liquidation charges |new| * cost at the NEXT bar; CARRY charges none.
        term = run(0.0, 1.0, terminal=True, config=ExecutionConfig(trade_lot_size=2, terminal_policy=TerminalPolicy.LIQUIDATE_AT_NEXT))
        self.assertAlmostEqual(term.exit_cost, 1.0 * 0.05 * 200.0)
        self.assertEqual(run(0.0, 1.0, terminal=True).exit_cost, 0.0)  # cfg is CARRY
        # delayed_close is honestly NOT a real executable fill; no fill prices.
        self.assertFalse(cfg.real_executable_fill_model)
        self.assertIsNone(ca.entry_fill_price)
        # quote_side: buy fills at ask, sell at bid, and IS a real executable fill model.
        qcfg = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE, trade_lot_size=1)
        q_now = MarketSnapshot(mid=100.0, best_bid=99.9, best_ask=100.1)
        q = simulate_transition(PositionState(position=0.0), 1.0, q_now, q_now, q_now, is_terminal=False, config=qcfg)
        self.assertTrue(qcfg.real_executable_fill_model)
        self.assertAlmostEqual(q.entry_fill_price, 100.1)  # buy at ask
        sell = simulate_transition(PositionState(position=1.0), 0.0, q_now, q_now, q_now, is_terminal=False, config=qcfg)
        self.assertAlmostEqual(sell.entry_fill_price, 99.9)  # closing the long sells at bid

    def test_execution_config_and_terminal_state_guards(self) -> None:
        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            ImpactModel,
            MarketSnapshot,
            PositionState,
            TerminalPolicy,
            fill_index,
            simulate_transition,
        )

        # Terminal liquidation flattens ALL held state, even on a hold-into-terminal (no stale bars/entry).
        out = simulate_transition(
            PositionState(position=1.0, bars_held=7, entry_price=100.0), 1.0,
            MarketSnapshot(mid=100.0), MarketSnapshot(mid=101.0, half_spread=0.05),
            MarketSnapshot(mid=102.0, half_spread=0.05), is_terminal=True,
            config=ExecutionConfig(terminal_policy=TerminalPolicy.LIQUIDATE_AT_NEXT),
        )
        self.assertEqual(out.next_state.position, 0.0)
        self.assertEqual(out.next_state.bars_held, 0)
        self.assertIsNone(out.next_state.entry_price)
        # CARRY keeps the held position and increments bars_held on a hold-into-terminal.
        carry = simulate_transition(
            PositionState(position=1.0, bars_held=7, entry_price=100.0), 1.0,
            MarketSnapshot(mid=100.0), MarketSnapshot(mid=101.0), MarketSnapshot(mid=102.0),
            is_terminal=True, config=ExecutionConfig(terminal_policy=TerminalPolicy.CARRY),
        )
        self.assertEqual(carry.next_state.position, 1.0)
        self.assertEqual(carry.next_state.bars_held, 8)
        # ExecutionConfig fails closed on invalid execution parameters.
        for kwargs in (
            {"latency_steps": -1},
            {"step_horizon": 0},
            {"trade_lot_size": 0},
            {"commission_per_share": -0.01},
            {"spread_multiplier": -1.0},
            # NaN/Inf must be rejected too: a bare `< 0` check passes NaN (every NaN comparison is False).
            {"commission_per_share": float("nan")},
            {"spread_multiplier": float("inf")},
            # Unknown fill level / terminal policy fail closed (not a confusing late crash).
            {"fill_level": "typo"},
            {"terminal_policy": "hold_forever"},
        ):
            with self.assertRaises(ValueError):
                ExecutionConfig(**kwargs)
        # A valid string fill_level is coerced to the enum (so callers may pass either).
        self.assertIs(ExecutionConfig(fill_level="quote_side").fill_level, FillLevel.QUOTE_SIDE)
        # ImpactModel self-validates: unknown kind (silent-impact-disable typo) and NaN/negative coef.
        for bad in ({"kind": "liner"}, {"kind": "linear", "coef_per_unit": -1.0}, {"kind": "linear", "coef_per_unit": float("nan")}):
            with self.assertRaises(ValueError):
                ImpactModel(**bad)
        # MarketSnapshot rejects a negative/NaN half_spread (negative cost would pay the agent to trade),
        # a non-finite mid, and an inverted quote.
        for bad in ({"mid": 100.0, "half_spread": -0.01}, {"mid": float("nan")}, {"mid": 100.0, "best_bid": 100.2, "best_ask": 99.8}):
            with self.assertRaises(ValueError):
                MarketSnapshot(**bad)
        # fill_index: negative latency clamps to the current bar (no pre-decision fill); horizon<=0 rejected.
        self.assertEqual(fill_index(10, step_horizon=5, latency_steps=-3), 10)
        with self.assertRaises(ValueError):
            fill_index(10, step_horizon=0, latency_steps=1)

    def test_execution_validation_hardening_followup(self) -> None:
        # Follow-up validation hardening (review of 0732733): type-safe bar/lot counts, impact_model
        # coercion, quote_side_plus_impact must carry real impact, positive/in-quote prices, and
        # PositionState invariants. None of these change a valid run; they only fail closed on bad input.
        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            ImpactModel,
            MarketSnapshot,
            PositionState,
            SymbolQuote,
            simulate_transition,
        )

        # (a) Integer-like bars/lot: reject bool and fractional floats; ACCEPT integer-valued floats and
        # coerce-store as int (so trade_scale et al. never see a fractional lot).
        for kwargs in (
            {"latency_steps": True}, {"step_horizon": True}, {"trade_lot_size": True},
            {"latency_steps": 0.9}, {"step_horizon": 1.5}, {"trade_lot_size": 1.5},
            {"step_horizon": float("nan")}, {"trade_lot_size": float("inf")},
        ):
            with self.assertRaises(ValueError):
                ExecutionConfig(**kwargs)
        coerced = ExecutionConfig(trade_lot_size=2.0, step_horizon=3.0, latency_steps=1.0)
        self.assertEqual((coerced.trade_lot_size, coerced.step_horizon, coerced.latency_steps), (2, 3, 1))
        for value in (coerced.trade_lot_size, coerced.step_horizon, coerced.latency_steps):
            self.assertIsInstance(value, int)
        self.assertEqual(coerced.trade_scale, 200.0)

        # (b) impact_model: a mapping is coerced to ImpactModel; a bare string (or other type) is rejected
        # at construction instead of crashing later on a missing .kind.
        mapped = ExecutionConfig(
            fill_level=FillLevel.QUOTE_SIDE_PLUS_IMPACT, impact_model={"kind": "linear", "coef_per_unit": 0.01}
        )
        self.assertIsInstance(mapped.impact_model, ImpactModel)
        self.assertEqual(mapped.impact_model.kind, "linear")
        for bad_impact in ("linear", 3, ["linear"]):
            with self.assertRaises(ValueError):
                ExecutionConfig(impact_model=bad_impact)

        # (c) quote_side_plus_impact must apply a positive linear impact, else it is indistinguishable
        # from quote_side yet claims to model impact.
        for bad in (ImpactModel(kind="none"), ImpactModel(kind="linear", coef_per_unit=0.0)):
            with self.assertRaises(ValueError):
                ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE_PLUS_IMPACT, impact_model=bad)

        # Granular fill-model properties (the report layer composes reportability from these + logs).
        proxy = ExecutionConfig(fill_level=FillLevel.DELAYED_CLOSE)
        self.assertEqual(
            (proxy.proxy_fill_model, proxy.uses_crossable_quote_fills, proxy.applies_implemented_impact, proxy.real_executable_fill_model),
            (True, False, False, False),
        )
        qside = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE)
        self.assertEqual(
            (qside.proxy_fill_model, qside.uses_crossable_quote_fills, qside.applies_implemented_impact, qside.real_executable_fill_model),
            (False, True, False, True),
        )
        self.assertEqual(
            (mapped.proxy_fill_model, mapped.uses_crossable_quote_fills, mapped.applies_implemented_impact, mapped.real_executable_fill_model),
            (False, True, True, True),
        )

        # MarketSnapshot: mid must be positive and inside [best_bid, best_ask] when both are present.
        for bad in (
            {"mid": 0.0}, {"mid": -1.0},
            {"mid": 200.0, "best_bid": 99.9, "best_ask": 100.1},  # above the ask
            {"mid": 50.0, "best_bid": 99.9, "best_ask": 100.1},  # below the bid
        ):
            with self.assertRaises(ValueError):
                MarketSnapshot(**bad)
        MarketSnapshot(mid=100.0, best_bid=99.9, best_ask=100.1)  # valid, no raise

        # SymbolQuote enforces the SAME mid-inside-quote invariant so _market() can't silently degrade a
        # malformed-but-present quote to MISSING_QUOTE; a valid quote round-trips through _market().
        with self.assertRaises(ValueError):
            SymbolQuote(symbol="X", mid=200.0, best_bid=99.9, best_ask=100.1)
        ok_quote = SymbolQuote(symbol="X", mid=100.0, best_bid=99.9, best_ask=100.1)
        snap = ok_quote._market()
        self.assertEqual((snap.mid, snap.best_bid, snap.best_ask), (100.0, 99.9, 100.1))

        # PositionState invariants: finite position, non-negative integer bars_held, positive entry_price,
        # and a flat (0) book carries no entry_price. Integer-valued bars_held floats coerce to int.
        for kwargs in (
            {"position": float("nan")},
            {"position": 1.0, "bars_held": -1},
            {"position": 1.0, "bars_held": 1.5},
            {"position": 1.0, "bars_held": True},
            {"position": 1.0, "entry_price": -5.0},
            {"position": 0.0, "entry_price": 100.0},  # flat must not carry an entry price
        ):
            with self.assertRaises(ValueError):
                PositionState(**kwargs)
        held = PositionState(position=1.0, bars_held=2.0, entry_price=100.0)
        self.assertEqual(held.bars_held, 2)
        self.assertIsInstance(held.bars_held, int)
        self.assertIsNone(PositionState(position=0.0).entry_price)

        # A quote-side transition to flat records the close-out price on the OUTCOME (entry_fill_price)
        # but leaves the flat next_state with entry_price=None (no stale exit price on a flat book).
        qcfg = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE)
        q_now = MarketSnapshot(mid=100.0, best_bid=99.9, best_ask=100.1)
        to_flat = simulate_transition(PositionState(position=1.0), 0.0, q_now, q_now, q_now, is_terminal=False, config=qcfg)
        self.assertAlmostEqual(to_flat.entry_fill_price, 99.9)  # sold the long at the bid
        self.assertEqual(to_flat.next_state.position, 0.0)
        self.assertIsNone(to_flat.next_state.entry_price)

    def test_execution_simulator_reproduces_intraday_reward(self) -> None:
        # Equivalence gate: delayed_close net_return must equal the intraday inline arithmetic exactly,
        # so the later intraday wiring is result-preserving.
        from rl_quant.execution import ExecutionConfig, MarketSnapshot, PositionState, TerminalPolicy, simulate_transition

        trade_lot_size, commission, extra = 3, 0.01, 0.005
        scale = trade_lot_size * 100.0
        cfg = ExecutionConfig(
            trade_lot_size=trade_lot_size, commission_per_share=commission, extra_cost_per_share=extra,
            terminal_policy=TerminalPolicy.LIQUIDATE_AT_NEXT,
        )
        mids = [100.0, 100.5, 99.5]
        spreads = [0.03, 0.04]
        for old in (-1.0, 0.0, 1.0):
            for new in (-1.0, 0.0, 1.0):
                for mid_now in mids:
                    for mid_fill in mids:
                        for mid_next in mids:
                            for hs_fill in spreads:
                                for hs_next in spreads:
                                    for terminal in (False, True):
                                        turnover = abs(new - old)
                                        expected = (
                                            old * (mid_fill - mid_now)
                                            + new * (mid_next - mid_fill)
                                            - turnover * (hs_fill + extra + commission)
                                        ) * scale
                                        if terminal and new != 0.0:
                                            expected -= abs(new) * (hs_next + extra + commission) * scale
                                        out = simulate_transition(
                                            PositionState(position=old),
                                            new,
                                            MarketSnapshot(mid=mid_now, half_spread=0.0),
                                            MarketSnapshot(mid=mid_fill, half_spread=hs_fill),
                                            MarketSnapshot(mid=mid_next, half_spread=hs_next),
                                            is_terminal=terminal,
                                            config=cfg,
                                        )
                                        self.assertAlmostEqual(out.net_return, expected, places=6)

    def test_transition_pnl_matches_inline_and_simulate(self) -> None:
        # transition_pnl is the single source of truth wired into the intraday env/eval/pretraining
        # sites. Prove it reproduces the inline arithmetic in scalar, vectorized, and 3x3-broadcast forms
        # (so the wiring is result-preserving), and equals simulate_transition for delayed_close.
        from rl_quant.execution import (
            ExecutionConfig,
            MarketSnapshot,
            PositionState,
            simulate_transition,
            transition_pnl,
        )

        scale, comm, extra = 200.0, 0.01, 0.005

        def inline(old, new, mn, mf, mx, hf, hn, term):
            r = (old * (mf - mn) + new * (mx - mf) - abs(new - old) * (hf + extra + comm)) * scale
            return r - (1.0 if term else 0.0) * abs(new) * (hn + extra + comm) * scale

        # Scalar: transition_pnl == inline, and == simulate_transition.net_return (delayed_close).
        cfg = ExecutionConfig(trade_lot_size=2, commission_per_share=comm, extra_cost_per_share=extra)
        for old in (-1.0, 0.0, 1.0):
            for new in (-1.0, 0.0, 1.0):
                for term in (False, True):
                    tp = transition_pnl(old, new, 100.0, 101.0, 99.0, 0.03, 0.04, term,
                                        trade_scale=scale, commission_per_share=comm, extra_cost_per_share=extra)
                    self.assertAlmostEqual(tp, inline(old, new, 100.0, 101.0, 99.0, 0.03, 0.04, term), places=6)
                    sim = simulate_transition(
                        PositionState(position=old), new,
                        MarketSnapshot(mid=100.0), MarketSnapshot(mid=101.0, half_spread=0.03),
                        MarketSnapshot(mid=99.0, half_spread=0.04), is_terminal=term, config=cfg,
                    )
                    self.assertAlmostEqual(sim.net_return, tp, places=6)

        # Vectorized (env path): long positions, float price/spread tensors, bool terminal -> elementwise match.
        old = torch.tensor([-1, 0, 1, 1], dtype=torch.long)
        new = torch.tensor([1, 1, 0, 1], dtype=torch.long)
        mn = torch.tensor([100.0, 100.0, 100.0, 100.0])
        mf = torch.tensor([100.5, 99.5, 101.0, 100.2])
        mx = torch.tensor([101.0, 99.0, 102.0, 100.4])
        hf = torch.tensor([0.03, 0.04, 0.03, 0.05])
        hn = torch.tensor([0.04, 0.05, 0.04, 0.06])
        term = torch.tensor([False, True, True, False])
        got = transition_pnl(old, new, mn, mf, mx, hf, hn, term, trade_scale=scale, commission_per_share=comm, extra_cost_per_share=extra)
        want = (old.float() * (mf - mn) + new.float() * (mx - mf) - (new - old).abs().float() * (hf + extra + comm)) * scale
        want = want - term.float() * new.abs().float() * (hn + extra + comm) * scale
        # atol is loose: folding the env's two-statement reward into one expression re-associates
        # float32 adds (~1e-5 noise on values ~100); any real operand error would be O(1)+.
        self.assertTrue(torch.allclose(got, want, atol=1e-3))

        # 3x3 broadcast (pretraining path): current [1,3,1], candidate [1,1,3], market [N,1,1] -> [N,3,3].
        positions = torch.tensor([-1.0, 0.0, 1.0])
        cur = positions.view(1, 3, 1)
        cand = positions.view(1, 1, 3)
        mn3, mf3, mx3 = mn.view(-1, 1, 1), mf.view(-1, 1, 1), mx.view(-1, 1, 1)
        hf3, hn3, term3 = hf.view(-1, 1, 1), hn.view(-1, 1, 1), term.view(-1, 1, 1)
        grid = transition_pnl(cur, cand, mn3, mf3, mx3, hf3, hn3, term3, trade_scale=scale, commission_per_share=comm, extra_cost_per_share=extra)
        self.assertEqual(tuple(grid.shape), (4, 3, 3))
        want_grid = (
            cur * (mf3 - mn3) + cand * (mx3 - mf3)
            - (cand - cur).abs() * (hf3 + extra + comm)
            - term3.float() * cand.abs() * (hn3 + extra + comm)
        ) * scale
        self.assertTrue(torch.allclose(grid, want_grid, atol=1e-3))

    def test_leg_level_action_transition(self) -> None:
        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            FillStatus,
            Holdings,
            LegSide,
            SymbolQuote,
            simulate_action_transition,
        )

        proxy = ExecutionConfig(fill_level=FillLevel.DELAYED_CLOSE, spread_multiplier=1.0)
        quote_cfg = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE)

        def q(sym, *, ret=0.0, lat=0.0, hs=0.0, bid=None, ask=None, mid=100.0):
            return SymbolQuote(symbol=sym, mid=mid, interval_return=ret, latency_return=lat,
                               half_spread=hs, best_bid=bid, best_ask=ask)

        cash = Holdings(())
        qqq = Holdings.single_slot("QQQ", 1.0)
        sqqq = Holdings.single_slot("SQQQ", 1.0)

        # cash -> QQQ: one BUY leg; RETURN-based gross = weight*interval_return; proxy cost = half_spread/mid bps.
        out = simulate_action_transition(cash, qqq, {"QQQ": q("QQQ", ret=0.02, hs=0.05, mid=100.0)}, proxy)
        self.assertEqual([(leg.symbol, leg.side) for leg in out.legs], [("QQQ", LegSide.BUY)])
        self.assertAlmostEqual(out.gross_mark_pnl, 0.02)
        self.assertAlmostEqual(out.legs[0].spread_bps, 5.0)  # 0.05 / 100 * 1e4
        self.assertAlmostEqual(out.realized_execution_cost, 1.0 * 5.0 / 1e4)
        self.assertAlmostEqual(out.net_pnl, 0.02 - 0.0005)
        self.assertIsNone(out.legs[0].fill_price)  # proxy fill -> no executable price
        self.assertFalse(out.real_executable_fill_model)
        # QQQ -> cash: one SELL leg.
        self.assertEqual(
            [(leg.symbol, leg.side) for leg in
             simulate_action_transition(qqq, cash, {"QQQ": q("QQQ", hs=0.05)}, proxy).legs],
            [("QQQ", LegSide.SELL)],
        )
        # QQQ -> SPY switch: two legs (sell QQQ + buy SPY), each on its own book, EXITS BEFORE ENTRIES.
        spy = Holdings.single_slot("SPY", 1.0)
        switch = simulate_action_transition(qqq, spy, {"QQQ": q("QQQ", hs=0.05), "SPY": q("SPY", ret=0.01, hs=0.05)}, proxy)
        self.assertEqual([(leg.symbol, leg.side) for leg in switch.legs], [("QQQ", LegSide.SELL), ("SPY", LegSide.BUY)])
        self.assertEqual(switch.next_state, spy)  # both legs filled -> executed == target
        # Buying inverse SQQQ fills at SQQQ's OWN ask (NOT "inverse -> sell"); no leverage multiplier on gross.
        inv = simulate_action_transition(cash, sqqq, {"SQQQ": q("SQQQ", ret=0.03, mid=20.0, bid=19.98, ask=20.02)}, quote_cfg)
        self.assertEqual(inv.legs[0].side, LegSide.BUY)
        self.assertAlmostEqual(inv.legs[0].fill_price, 20.02)  # SQQQ ask, not QQQ / not a sell
        self.assertAlmostEqual(inv.gross_mark_pnl, 0.03)  # weight * its own return, no leverage mult
        self.assertTrue(inv.real_executable_fill_model)
        self.assertEqual(
            simulate_action_transition(sqqq, cash, {"SQQQ": q("SQQQ", mid=20.0, bid=19.98, ask=20.02)}, quote_cfg).legs[0].fill_price,
            19.98,  # closing the long sells at the bid
        )
        # Missing quote at quote-side -> MISSING_QUOTE leg + non-(real-executable) + warning; proxy tolerates it.
        # Fail-closed: the unfilled trade does NOT teleport the book to the target -- it stays flat and earns
        # nothing on the would-be position (no interval P&L on an unfilled target).
        miss = simulate_action_transition(cash, qqq, {"QQQ": q("QQQ", ret=0.02)}, quote_cfg)
        self.assertEqual(miss.legs[0].fill_status, FillStatus.MISSING_QUOTE)
        self.assertFalse(miss.real_executable_fill_model)
        self.assertIn("missing_quote:QQQ", miss.warnings)
        self.assertEqual(miss.next_state, cash)  # blocked: no fill -> keep prior (cash) holdings
        self.assertAlmostEqual(miss.gross_mark_pnl, 0.0)  # earns nothing on the unfilled target
        self.assertEqual(miss.realized_execution_cost, 0.0)
        self.assertFalse(miss.execution_complete)  # the requested buy did not fill
        self.assertTrue(miss.valuation_complete)  # nothing non-zero was left unvalued (ended in cash)
        self.assertEqual(
            simulate_action_transition(cash, qqq, {"QQQ": q("QQQ", ret=0.02)}, proxy).legs[0].fill_status,
            FillStatus.FILLED,
        )
        # Missing quote entirely (symbol absent from the market map) also blocks the trade, keeping cash.
        absent = simulate_action_transition(cash, qqq, {}, quote_cfg)
        self.assertEqual(absent.next_state, cash)
        self.assertEqual(len(absent.legs), 0)
        self.assertFalse(absent.real_executable_fill_model)
        self.assertIn("missing_quote:QQQ", absent.warnings)
        # Hold (same action) -> no legs, no cost; held weight earns latency + interval return.
        hold = simulate_action_transition(qqq, qqq, {"QQQ": q("QQQ", ret=0.02, lat=0.01, hs=0.05)}, proxy)
        self.assertEqual(len(hold.legs), 0)
        self.assertEqual(hold.realized_execution_cost, 0.0)
        self.assertAlmostEqual(hold.gross_mark_pnl, 0.02 + 0.01)

    def test_leg_level_failclosed_and_helper_strictness(self) -> None:
        # Follow-up hardening (review of 0b3dd15): leg-level fail-closed semantics (no free terminal
        # liquidation, Holdings validation) + public fill-timing helpers reject fractional inputs +
        # numeric fields coerce-and-store. The leg layer is still unwired, so none of this changes a reward.
        import torch as _torch

        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            Holdings,
            MarketSnapshot,
            PositionState,
            SymbolQuote,
            fill_index,
            fill_indices,
            simulate_action_transition,
        )

        def q(sym, *, ret=0.0, lat=0.0, hs=0.0, bid=None, ask=None, mid=100.0):
            return SymbolQuote(symbol=sym, mid=mid, interval_return=ret, latency_return=lat,
                               half_spread=hs, best_bid=bid, best_ask=ask)

        quote_cfg = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE, terminal_policy="liquidate_at_next")
        qqq = Holdings.single_slot("QQQ", 1.0)

        # Terminal liquidation with a MISSING quote must NOT flatten the book for free: the holding stays
        # and the outcome is flagged with a terminal_missing_quote warning.
        term_block = simulate_action_transition(qqq, qqq, {}, quote_cfg, is_terminal=True)
        self.assertEqual(term_block.next_state, qqq)  # could not liquidate -> still held
        self.assertFalse(term_block.real_executable_fill_model)
        self.assertFalse(term_block.execution_complete)  # liquidation leg could not fill
        self.assertIn("terminal_missing_quote:QQQ", term_block.warnings)
        # With a real quote, terminal liquidation flattens to cash and books the exit leg + its cost.
        term_ok = simulate_action_transition(
            qqq, qqq, {"QQQ": q("QQQ", mid=100.0, bid=99.9, ask=100.1)}, quote_cfg, is_terminal=True
        )
        self.assertEqual(term_ok.next_state, Holdings(()))
        self.assertGreater(term_ok.realized_execution_cost, 0.0)

        # A HELD position (no trade) whose symbol has no quote cannot be valued: it must be flagged
        # non-(real-executable) with a warning, NOT silently credited a 0 return while staying "real".
        held_no_quote = simulate_action_transition(qqq, qqq, {}, quote_cfg)
        self.assertFalse(held_no_quote.real_executable_fill_model)
        self.assertIn("missing_quote:QQQ", held_no_quote.warnings)
        self.assertEqual(held_no_quote.next_state, qqq)  # still held (unvalued, not flattened)
        self.assertEqual(held_no_quote.gross_mark_pnl, 0.0)  # cannot value it -> 0, but flagged non-real
        # Decomposed status: could not VALUE the held position, but nothing FAILED to execute (no trade).
        self.assertFalse(held_no_quote.valuation_complete)
        self.assertTrue(held_no_quote.execution_complete)
        # A held position WITH a quote is valued normally and stays real-executable.
        held_ok = simulate_action_transition(qqq, qqq, {"QQQ": q("QQQ", ret=0.02, lat=0.01, bid=99.9, ask=100.1)}, quote_cfg)
        self.assertTrue(held_ok.real_executable_fill_model)
        self.assertEqual(held_ok.warnings, ())
        self.assertAlmostEqual(held_ok.gross_mark_pnl, 0.03)
        self.assertTrue(held_ok.valuation_complete)
        self.assertTrue(held_ok.execution_complete)

        # Public integer-like validators are exported so the env (and other callers) enforce the same
        # bar/lot rules as ExecutionConfig instead of int()-truncating a fractional config value.
        from rl_quant.execution import require_nonnegative_int, require_positive_int

        self.assertEqual(require_positive_int("h", 3), 3)
        self.assertEqual(require_nonnegative_int("l", 0), 0)
        for bad in (1.9, True, -1, "x"):
            with self.assertRaises(ValueError):
                require_positive_int("h", bad)
        for bad in (0.5, True, -1):
            with self.assertRaises(ValueError):
                require_nonnegative_int("l", bad)

        # Holdings fails closed: duplicate symbol, non-finite weight, explicit CASH; ~0 weights are dropped.
        for bad in ((("QQQ", 0.5), ("QQQ", 0.7)), (("QQQ", float("nan")),), (("CASH", 1.0),)):
            with self.assertRaises(ValueError):
                Holdings(bad)
        self.assertEqual(Holdings((("QQQ", 1.0), ("SPY", 0.0))).symbols(), ("QQQ",))  # ~0 dropped

        # Public fill-timing helpers reject fractional/bool bar+latency (not just ExecutionConfig), and the
        # vectorized helper requires an integer index tensor. Valid integer calls still agree element-wise.
        for kwargs in ({"step_horizon": 1.5, "latency_steps": 1}, {"step_horizon": 5, "latency_steps": 0.5},
                       {"step_horizon": True, "latency_steps": 1}):
            with self.assertRaises(ValueError):
                fill_index(10, **kwargs)
            with self.assertRaises(ValueError):
                fill_indices(_torch.arange(4), **kwargs)
        with self.assertRaises(ValueError):
            fill_indices(_torch.arange(4, dtype=_torch.float32), step_horizon=5, latency_steps=1)
        # Small/unsigned integer dtypes overflow as bar indices and are rejected; int32/int64 are allowed.
        for bad_dtype in (_torch.uint8, _torch.int8, _torch.int16):
            with self.assertRaises(ValueError):
                fill_indices(_torch.arange(4, dtype=bad_dtype), step_horizon=5, latency_steps=1)
        for ok_dtype in (_torch.int32, _torch.int64):
            self.assertEqual(fill_indices(_torch.arange(4, dtype=ok_dtype), step_horizon=5, latency_steps=1).dtype, ok_dtype)
        # Scalar fill_index also validates now_index: reject fractional/bool and (critically) NEGATIVE bar
        # indices, which would otherwise index from the end of the array (PyTorch negative-indexing footgun).
        for bad_now in (1.5, True, -1):
            with self.assertRaises(ValueError):
                fill_index(bad_now, step_horizon=5, latency_steps=1)
        self.assertEqual(fill_index(0, step_horizon=5, latency_steps=1), 1)  # zero is a valid first bar
        self.assertEqual(
            fill_indices(_torch.arange(6), step_horizon=5, latency_steps=2).tolist(),
            [fill_index(i, step_horizon=5, latency_steps=2) for i in range(6)],
        )

        # Numeric fields coerce-and-store (a value that only validated but stayed a string would later break
        # arithmetic); bool is rejected everywhere numeric; best_bid/best_ask must be positive.
        cfg = ExecutionConfig(commission_per_share="0.01", extra_cost_per_share="0.02")
        self.assertIsInstance(cfg.commission_per_share, float)
        self.assertAlmostEqual(cfg.commission_per_share, 0.01)
        self.assertEqual(MarketSnapshot(mid="100.0").mid, 100.0)
        self.assertEqual(PositionState(position="1.0", entry_price="100.0").position, 1.0)
        for bad in ({"commission_per_share": True}, {"spread_multiplier": True}):
            with self.assertRaises(ValueError):
                ExecutionConfig(**bad)
        for bad in (
            {"mid": 50.0, "best_bid": -1.0, "best_ask": 100.0},
            {"mid": 50.0, "best_bid": 0.0, "best_ask": 100.0},
            {"mid": True},
        ):
            with self.assertRaises(ValueError):
                MarketSnapshot(**bad)

    def test_switch_fill_policy_atomic_vs_independent(self) -> None:
        # PR4: opt-in ATOMIC_SWITCH all-or-nothing partial-switch policy on the (still-unwired) leg layer.
        # Default stays INDEPENDENT_LEGS so no existing behavior/reward changes.
        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            Holdings,
            LegSide,
            SwitchFillPolicy,
            SymbolQuote,
            simulate_action_transition,
        )

        def q(sym, *, ret=0.0, lat=0.0, mid=100.0, bid=None, ask=None):
            return SymbolQuote(symbol=sym, mid=mid, interval_return=ret, latency_return=lat, best_bid=bid, best_ask=ask)

        a = Holdings.single_slot("A", 1.0)
        b = Holdings.single_slot("B", 1.0)

        # Default is INDEPENDENT_LEGS; string coerces to the enum; an unknown value fails closed.
        self.assertIs(ExecutionConfig().switch_fill_policy, SwitchFillPolicy.INDEPENDENT_LEGS)
        self.assertIs(
            ExecutionConfig(switch_fill_policy="atomic_switch").switch_fill_policy, SwitchFillPolicy.ATOMIC_SWITCH
        )
        with self.assertRaises(ValueError):
            ExecutionConfig(switch_fill_policy="atomic")

        indep = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE, switch_fill_policy=SwitchFillPolicy.INDEPENDENT_LEGS)
        atomic = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE, switch_fill_policy=SwitchFillPolicy.ATOMIC_SWITCH)

        # A -> B where B has no bid/ask (B buy cannot fill at quote_side) but A's sell can fill.
        partial = {
            "A": q("A", ret=0.02, lat=0.01, mid=100.0, bid=99.9, ask=100.1),
            "B": q("B", ret=0.05, mid=50.0),  # missing quotes -> unfillable buy
        }
        # INDEPENDENT: A sells (fills), B buy blocked -> stranded in cash (the non-conserving partial fill).
        out_indep = simulate_action_transition(a, b, partial, indep)
        self.assertEqual(out_indep.next_state, Holdings(()))
        self.assertFalse(out_indep.real_executable_fill_model)
        # ATOMIC: B buy unfillable -> NOTHING executes, keep A; A still earns its latency+interval return.
        out_atomic = simulate_action_transition(a, b, partial, atomic)
        self.assertEqual(out_atomic.next_state, a)
        self.assertFalse(out_atomic.real_executable_fill_model)
        self.assertFalse(out_atomic.execution_complete)
        self.assertEqual(len(out_atomic.legs), 0)
        self.assertEqual(out_atomic.realized_execution_cost, 0.0)
        self.assertIn("missing_quote:B", out_atomic.warnings)
        self.assertIn("atomic_switch_blocked", out_atomic.warnings)
        self.assertAlmostEqual(out_atomic.gross_mark_pnl, 1.0 * 0.01 + 1.0 * 0.02)  # held A: latency + interval

        # When ALL legs fill, ATOMIC == INDEPENDENT: both reach target B with the normal sell+buy legs.
        full = {
            "A": q("A", ret=0.02, mid=100.0, bid=99.9, ask=100.1),
            "B": q("B", ret=0.05, mid=50.0, bid=49.95, ask=50.05),
        }
        done = simulate_action_transition(a, b, full, atomic)
        self.assertEqual(done.next_state, b)
        self.assertTrue(done.real_executable_fill_model)
        self.assertTrue(done.execution_complete)
        self.assertEqual({(leg.symbol, leg.side) for leg in done.legs}, {("A", LegSide.SELL), ("B", LegSide.BUY)})

    def test_weight_execution_cost_bps(self) -> None:
        # PR5: bps-denominated fee/impact for the return/weight-based leg layer (distinct from the scalar
        # per-share dollar fields). Default is zero -> leg cost stays spread-only (existing behavior).
        from rl_quant.execution import (
            ExecutionConfig,
            FillLevel,
            FillStatus,
            Holdings,
            SymbolQuote,
            WeightExecutionCostConfig,
            simulate_action_transition,
        )

        cash = Holdings(())
        qqq_full = Holdings.single_slot("QQQ", 1.0)
        qqq_half = Holdings.single_slot("QQQ", 0.5)

        def q(*, hs=0.05, ret=0.0, mid=100.0, bid=None, ask=None):
            return SymbolQuote(symbol="QQQ", mid=mid, interval_return=ret, half_spread=hs, best_bid=bid, best_ask=ask)

        # Default weight_cost -> spread-only: fee/impact zero, total == spread, cost unchanged.
        base = ExecutionConfig(fill_level=FillLevel.DELAYED_CLOSE, spread_multiplier=1.0)
        out0 = simulate_action_transition(cash, qqq_full, {"QQQ": q(ret=0.02)}, base)
        leg0 = out0.legs[0]
        self.assertAlmostEqual(leg0.spread_bps, 5.0)  # 0.05/100*1e4
        self.assertEqual((leg0.fee_bps, leg0.impact_bps), (0.0, 0.0))
        self.assertAlmostEqual(leg0.total_cost_bps, 5.0)
        self.assertAlmostEqual(out0.realized_execution_cost, 1.0 * 5.0 / 1e4)

        # fee + linear impact: total = spread + fee + coef*traded; cost = traded * total / 1e4.
        cost_cfg = ExecutionConfig(
            fill_level=FillLevel.DELAYED_CLOSE, spread_multiplier=1.0,
            weight_cost=WeightExecutionCostConfig(fee_bps=2.0, impact_kind="linear_bps", linear_impact_bps_per_weight=3.0),
        )
        full = simulate_action_transition(cash, qqq_full, {"QQQ": q(ret=0.02)}, cost_cfg)
        legf = full.legs[0]
        self.assertAlmostEqual(legf.fee_bps, 2.0)
        self.assertAlmostEqual(legf.impact_bps, 3.0)  # 3.0 bps/weight * 1.0 traded
        self.assertAlmostEqual(legf.total_cost_bps, 10.0)
        self.assertAlmostEqual(full.realized_execution_cost, 1.0 * 10.0 / 1e4)
        self.assertAlmostEqual(full.net_pnl, 0.02 - 0.001)

        # Linear impact is size-dependent: a half-weight trade pays half the impact bps.
        half = simulate_action_transition(cash, qqq_half, {"QQQ": q(ret=0.02)}, cost_cfg)
        legh = half.legs[0]
        self.assertAlmostEqual(legh.impact_bps, 1.5)  # 3.0 * 0.5
        self.assertAlmostEqual(legh.total_cost_bps, 5.0 + 2.0 + 1.5)
        self.assertAlmostEqual(half.realized_execution_cost, 0.5 * 8.5 / 1e4)

        # A blocked (MISSING_QUOTE) leg charges NO fee/impact (it did not execute).
        quote_cost = ExecutionConfig(
            fill_level=FillLevel.QUOTE_SIDE,
            weight_cost=WeightExecutionCostConfig(fee_bps=2.0, impact_kind="linear_bps", linear_impact_bps_per_weight=3.0),
        )
        blocked = simulate_action_transition(cash, qqq_full, {"QQQ": q(ret=0.02)}, quote_cost)  # no bid/ask
        self.assertEqual(blocked.legs[0].fill_status, FillStatus.MISSING_QUOTE)
        self.assertEqual((blocked.legs[0].fee_bps, blocked.legs[0].impact_bps), (0.0, 0.0))
        self.assertEqual(blocked.realized_execution_cost, 0.0)

        # WeightExecutionCostConfig validation + mapping coercion on ExecutionConfig.
        for bad in ({"impact_kind": "quadratic"}, {"fee_bps": -1.0}, {"linear_impact_bps_per_weight": -1.0}):
            with self.assertRaises(ValueError):
                WeightExecutionCostConfig(**bad)
        self.assertAlmostEqual(ExecutionConfig(weight_cost={"fee_bps": 2.0}).weight_cost.fee_bps, 2.0)
        with self.assertRaises(ValueError):
            ExecutionConfig(weight_cost="bad")

        # impact axes are separate: SCALAR impact (impact_model, quote_side_plus_impact) is NOT the LEG
        # impact (weight_cost). A quote_side_plus_impact config with default weight_cost charges ZERO leg
        # impact, so the leg outcome's impact_applied must be False (closes the "claims impact, charges none"
        # hazard). The leg path's impact_applied tracks weight_cost, independent of fill_level.
        qspi = ExecutionConfig(
            fill_level=FillLevel.QUOTE_SIDE_PLUS_IMPACT,
            impact_model={"kind": "linear", "coef_per_unit": 0.01},  # scalar impact present...
        )
        self.assertTrue(qspi.applies_implemented_impact)  # ...scalar axis on
        self.assertFalse(qspi.applies_weight_impact)  # ...but leg axis off (weight_cost default zero)
        self.assertFalse(ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE).applies_weight_impact)
        with_leg_impact = ExecutionConfig(
            fill_level=FillLevel.QUOTE_SIDE,
            weight_cost=WeightExecutionCostConfig(impact_kind="linear_bps", linear_impact_bps_per_weight=3.0),
        )
        self.assertTrue(with_leg_impact.applies_weight_impact)
        out_no_impact = simulate_action_transition(
            cash, qqq_full, {"QQQ": q(bid=99.9, ask=100.1)}, qspi
        )
        self.assertFalse(out_no_impact.impact_applied)  # crossable + valued + executed, but NO leg impact
        self.assertTrue(out_no_impact.real_executable_fill_model)  # impact is a separate axis from realness
        out_impact = simulate_action_transition(
            cash, qqq_full, {"QQQ": q(bid=99.9, ask=100.1)}, with_leg_impact
        )
        self.assertTrue(out_impact.impact_applied)
        # Transition-ACTUAL (reorg review): with impact CONFIGURED, a transition that charges no impact must
        # still report impact_applied=False -- it was config-level (always True for with_leg_impact) before.
        out_no_trade = simulate_action_transition(  # prev == target -> nothing trades -> no impact charged
            qqq_full, qqq_full, {"QQQ": q(bid=99.9, ask=100.1)}, with_leg_impact
        )
        self.assertFalse(out_no_trade.impact_applied)
        out_blocked = simulate_action_transition(cash, qqq_full, {}, with_leg_impact)  # buy leg blocked (no quote)
        self.assertFalse(out_blocked.impact_applied)  # leg never filled -> no impact, despite impact config

        # Leg + outcome invariants are enforced at construction.
        from rl_quant.execution import ActionTransitionOutcome, ExecutionLeg, FillStatus, Holdings, LegSide

        with self.assertRaises(ValueError):  # total_cost_bps must equal spread+fee+impact
            ExecutionLeg(symbol="X", side=LegSide.BUY, traded_weight=1.0, mark_before=0.0, mid_at_fill=100.0,
                         fill_price=None, spread_bps=5.0, fill_status=FillStatus.FILLED, total_cost_bps=4.0)
        with self.assertRaises(ValueError):  # an unfilled leg must not carry a fill_price
            ExecutionLeg(symbol="X", side=LegSide.BUY, traded_weight=1.0, mark_before=0.0, mid_at_fill=100.0,
                         fill_price=100.1, spread_bps=0.0, fill_status=FillStatus.MISSING_QUOTE)
        # A FILLED proxy leg with fill_price=None is VALID (the reviewer's filled=>price invariant is wrong).
        ExecutionLeg(symbol="X", side=LegSide.BUY, traded_weight=1.0, mark_before=0.0, mid_at_fill=100.0,
                     fill_price=None, spread_bps=5.0, fill_status=FillStatus.FILLED, total_cost_bps=5.0)
        with self.assertRaises(ValueError):  # net_pnl must equal gross - cost
            ActionTransitionOutcome(legs=(), old_position_latency_pnl=0.0, new_position_interval_pnl=0.0,
                                    gross_mark_pnl=1.0, realized_execution_cost=0.1, net_pnl=0.5,
                                    next_state=Holdings(()), real_executable_fill_model=False)
        with self.assertRaises(ValueError):  # real implies valuation_complete and execution_complete
            ActionTransitionOutcome(legs=(), old_position_latency_pnl=0.0, new_position_interval_pnl=0.0,
                                    gross_mark_pnl=0.0, realized_execution_cost=0.0, net_pnl=0.0,
                                    next_state=Holdings(()), real_executable_fill_model=True, execution_complete=False)

    def test_weight_transition_cost_bps_matches_engine(self) -> None:
        # PR-3: the vectorized weight_transition_cost_bps (used by the shadow env) must equal the per-leg
        # WeightExecutionCostConfig cost the dataclass engine charges, in bps (= 1e4 * realized_execution_cost).
        from rl_quant.execution import (
            ExecutionConfig, FillLevel, Holdings, SymbolQuote, WeightExecutionCostConfig,
            simulate_action_transition, weight_transition_cost_bps,
        )

        wc = WeightExecutionCostConfig(fee_bps=2.0)
        cfg = ExecutionConfig(fill_level=FillLevel.QUOTE_SIDE, weight_cost=wc)

        def zq(sym: str) -> SymbolQuote:  # zero spread -> total cost is fee only (isolates the fee term)
            return SymbolQuote(symbol=sym, mid=100.0, best_bid=100.0, best_ask=100.0)

        out_buy = simulate_action_transition(Holdings(()), Holdings((("QQQ", 1.0),)), {"QQQ": zq("QQQ")}, cfg)
        helper_buy = weight_transition_cost_bps(torch.tensor([0.0]), torch.tensor([1.0]), weight_cost=wc)
        self.assertAlmostEqual(float(helper_buy[0]), 1e4 * out_buy.realized_execution_cost, places=6)

        out_sw = simulate_action_transition(
            Holdings((("QQQ", 1.0),)), Holdings((("SPY", 0.5),)), {"QQQ": zq("QQQ"), "SPY": zq("SPY")}, cfg
        )
        helper_sw = weight_transition_cost_bps(torch.tensor([1.0]), torch.tensor([0.5]), weight_cost=wc)
        self.assertAlmostEqual(float(helper_sw[0]), 1e4 * out_sw.realized_execution_cost, places=6)
        self.assertAlmostEqual(float(helper_sw[0]), 3.0, places=6)  # (1.0 + 0.5) * 2 bps

    def test_minute_to_hour_execution_shadow_reward_side_channel(self) -> None:
        # PR-3: execution_env_reward_shadow ON computes a weight-bps execution reward/cost ALONGSIDE the legacy
        # reward (logged in the step dict) but leaves the training `rewards` byte-identical -- replay stores only
        # declared fields, so the shadow never reaches training. Default OFF emits no shadow keys.
        split = HourFromMinuteDataSplit(
            name="train",
            decision_timestamps=[f"2026-01-02T1{h}:30:00+00:00" for h in range(4)],
            next_timestamps=[f"2026-01-02T1{h + 1}:30:00+00:00" for h in range(4)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((4, 1, 1, 1)), minute_mask=torch.ones((4, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((4, 1, 1)),
            action_returns=torch.tensor([[0.0, 0.10], [0.0, -0.20], [0.0, 0.30], [0.0, 0.0]]),
            action_valid_mask=torch.ones((4, 2), dtype=torch.bool), label_valid_mask=torch.ones((4, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3]), valid_index_mask=torch.tensor([True, True, True, True]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        module = __import__("rl_quant.minute_to_hour_transformer",
                            fromlist=["VectorizedMinuteToHourEnv", "MinuteToHourEnvConfig"])

        def run(shadow: bool, action: int) -> dict:
            env = module.VectorizedMinuteToHourEnv(
                split, module.MinuteToHourEnvConfig(num_envs=1, episode_length=10, initial_action=0,
                                                    execution_env_reward_shadow=shadow), torch.device("cpu"))
            env.indices[:] = 0
            env.entry_index[:] = 0
            return env.step(torch.tensor([action], dtype=torch.long))

        off, on = run(False, 1), run(True, 1)  # switch CASH->QQQ
        self.assertTrue(torch.equal(off["rewards"], on["rewards"]))  # training reward byte-identical
        self.assertNotIn("execution_env_reward_shadow", off)
        for key in ("execution_env_reward_shadow", "execution_cost_bps_shadow", "reward_delta_shadow", "cost_delta_shadow"):
            self.assertIn(key, on)
        self.assertTrue(torch.isfinite(on["execution_cost_bps_shadow"]).all())
        self.assertGreater(float(on["execution_cost_bps_shadow"][0]), 0.0)  # a real switch trades
        hold = run(True, 0)  # CASH -> CASH: no trade -> zero shadow execution cost
        self.assertEqual(float(hold["execution_cost_bps_shadow"][0]), 0.0)

    def test_minute_to_hour_eval_applies_cash_idle_penalty(self) -> None:
        # Review #5 fix (eval-through-shared-primitive): evaluate_minute_to_hour_policy now applies the env's
        # cash_idle_penalty_bps via the SHARED transition_trade_cost_bps -- it omitted it before (a latent
        # drift vs the training reward). An always-cash policy's reward drops when the penalty is nonzero;
        # with penalty 0 it is byte-identical to the prior behaviour (no cost on zero-return cash holds).
        from rl_quant.minute_to_hour_transformer import HourFromMinuteDataSplit
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        split = HourFromMinuteDataSplit(
            name="eval",
            decision_timestamps=[f"2026-01-02T1{h}:30:00+00:00" for h in range(4)],
            next_timestamps=[f"2026-01-02T1{h + 1}:30:00+00:00" for h in range(4)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((4, 1, 1, 1)), minute_mask=torch.ones((4, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((4, 1, 1)), action_returns=torch.zeros((4, 2)),
            action_valid_mask=torch.ones((4, 2), dtype=torch.bool), label_valid_mask=torch.ones((4, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3]), valid_index_mask=torch.tensor([True, True, True, True]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        always_cash = _ConstantActionModel(2, 0)  # holds CASH every row
        device = torch.device("cpu")
        free = evaluate_minute_to_hour_policy(split, always_cash, device=device, cash_idle_penalty_bps=0.0)
        penalised = evaluate_minute_to_hour_policy(split, always_cash, device=device, cash_idle_penalty_bps=100.0)
        self.assertEqual(free.total_reward_bps, 0.0)  # zero returns, no trade, no penalty -> zero
        self.assertLess(penalised.total_reward_bps, free.total_reward_bps)  # cash-idle penalty now charged in eval

    def test_transition_cost_breakdown_table_and_env_reward_consistency(self) -> None:
        # Locks the shared transition-cost semantics (review #2/#3): the breakdown separates leg/execution cost
        # from the behavioural switch-penalty regularizer and the cash-idle penalty, and the env reward uses
        # exactly trade_cost_bps (= leg + switch) + cash_idle. Since env + eval call the SAME primitive, this
        # also pins their agreement.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import (
            MinuteToHourEnvConfig, VectorizedMinuteToHourEnv, transition_trade_cost_bps,
        )

        cons = dataclasses.replace(
            default_minute_to_hour_constraints(), one_way_cost_bps=10.0, extra_switch_penalty_bps=5.0, cash_index=0
        )

        def bd(prev: int, act: int, cash_idle: float = 0.0):
            return transition_trade_cost_bps(
                torch.tensor([prev]), torch.tensor([act]), constraints=cons, cash_idle_penalty_bps=cash_idle
            )

        cash_hold = bd(0, 0, cash_idle=50.0)  # CASH -> CASH: no trade; only cash-idle charged
        self.assertEqual(float(cash_hold.legs[0]), 0.0)
        self.assertEqual(float(cash_hold.leg_cost_bps[0]), 0.0)
        self.assertEqual(float(cash_hold.switch_penalty_bps[0]), 0.0)
        self.assertEqual(float(cash_hold.cash_idle_bps[0]), 50.0)

        enter = bd(0, 1, cash_idle=50.0)  # CASH -> asset 1: a switch; leg cost > 0; switch penalty once; no idle
        self.assertEqual(float(enter.legs[0]), 1.0)  # entering from cash = one buy leg
        self.assertEqual(float(enter.leg_cost_bps[0]), 10.0)  # legs * one_way_cost_bps
        self.assertEqual(float(enter.switch_penalty_bps[0]), 5.0)
        self.assertEqual(float(enter.cash_idle_bps[0]), 0.0)
        self.assertAlmostEqual(
            float(enter.trade_cost_bps[0]), float(enter.leg_cost_bps[0]) + float(enter.switch_penalty_bps[0]), places=6
        )

        held = bd(1, 1, cash_idle=50.0)  # hold asset 1: no switch, no idle
        self.assertEqual(float(held.switch_penalty_bps[0]), 0.0)
        self.assertEqual(float(held.cash_idle_bps[0]), 0.0)

        exit_ = bd(1, 0, cash_idle=50.0)  # asset 1 -> CASH: a switch (sell leg + penalty) AND cash idle charged
        self.assertEqual(float(exit_.legs[0]), 1.0)
        self.assertEqual(float(exit_.leg_cost_bps[0]), 10.0)
        self.assertEqual(float(exit_.switch_penalty_bps[0]), 5.0)
        self.assertEqual(float(exit_.cash_idle_bps[0]), 50.0)

        # ETF<->ETF (1 -> 2, both non-cash) leg count honours count_etf_to_etf_as_two_legs.
        two = dataclasses.replace(cons, count_etf_to_etf_as_two_legs=True)
        one = dataclasses.replace(cons, count_etf_to_etf_as_two_legs=False)
        legs_two = float(transition_trade_cost_bps(
            torch.tensor([1]), torch.tensor([2]), constraints=two, cash_idle_penalty_bps=0.0).legs[0])
        legs_one = float(transition_trade_cost_bps(
            torch.tensor([1]), torch.tensor([2]), constraints=one, cash_idle_penalty_bps=0.0).legs[0])
        self.assertEqual(legs_two, 2.0)  # sell ETF1 + buy ETF2 counted as two legs
        self.assertEqual(legs_one, 1.0)  # counted as a single switch leg

        # The env reward uses exactly (trade_cost_bps + cash_idle_bps) from the SAME primitive.
        split = HourFromMinuteDataSplit(
            name="t", decision_timestamps=[f"2026-01-02T1{h}:30:00+00:00" for h in range(3)],
            next_timestamps=[f"2026-01-02T1{h + 1}:30:00+00:00" for h in range(3)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((3, 1, 1, 1)), minute_mask=torch.ones((3, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((3, 1, 1)), action_returns=torch.tensor([[0.0, 0.10], [0.0, 0.0], [0.0, 0.0]]),
            action_valid_mask=torch.ones((3, 2), dtype=torch.bool), label_valid_mask=torch.ones((3, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2]), valid_index_mask=torch.tensor([True, True, True]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        env = VectorizedMinuteToHourEnv(
            split, MinuteToHourEnvConfig(num_envs=1, episode_length=10, initial_action=0,
                                         cash_idle_penalty_bps=50.0, constraints=cons), torch.device("cpu")
        )
        env.indices[:] = 0
        env.entry_index[:] = 0
        out = env.step(torch.tensor([1], dtype=torch.long))  # CASH -> QQQ, raw return 0.10
        b = bd(0, 1, cash_idle=50.0)  # cash_idle_bps is 0 here (QQQ is not cash) -- idle only charged on cash
        self.assertEqual(float(b.cash_idle_bps[0]), 0.0)
        expected = 0.10 * 10_000.0 - (float(b.trade_cost_bps[0]) + float(b.cash_idle_bps[0])) * 10_000.0 / 10_000.0
        self.assertAlmostEqual(float(out["rewards"][0]), expected, places=4)

    def test_minute_to_hour_env_rejects_non_cash_cash_index(self) -> None:
        # Review #2: cash_index must point to a CASH action, not merely be in range -- otherwise the wrong
        # action gets the cash-idle penalty / zero shadow exposure / label fallback. Out-of-range and
        # in-range-but-not-cash both fail closed; the correct cash index constructs fine.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.minute_to_hour_transformer import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv

        def split() -> HourFromMinuteDataSplit:
            return HourFromMinuteDataSplit(
                name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
                next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, 2)),
                action_valid_mask=torch.ones((2, 2), dtype=torch.bool), label_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                valid_start_indices=torch.tensor([0, 1]), valid_index_mask=torch.tensor([True, True]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        base = default_minute_to_hour_constraints()
        device = torch.device("cpu")
        # cash_index=0 -> "CASH": constructs fine.
        VectorizedMinuteToHourEnv(split(), MinuteToHourEnvConfig(num_envs=1, episode_length=5, constraints=base), device)
        # cash_index=1 -> "QQQ" (in range but not cash): fail closed.
        with self.assertRaises(ValueError):
            VectorizedMinuteToHourEnv(
                split(), MinuteToHourEnvConfig(num_envs=1, episode_length=5, initial_action=1,
                                               constraints=dataclasses.replace(base, cash_index=1)), device)
        # cash_index out of range: fail closed.
        with self.assertRaises(ValueError):
            VectorizedMinuteToHourEnv(
                split(), MinuteToHourEnvConfig(num_envs=1, episode_length=5,
                                               constraints=dataclasses.replace(base, cash_index=5)), device)

    def test_minute_to_hour_cash_index_validator_and_action_dtype_guards(self) -> None:
        # Review #6 (afdf0a6): the SHARED cash-index validator fails closed identically in the env AND the
        # evaluator (the eval previously could silently price the cash-idle penalty / cash fallback against the
        # wrong action); it rejects wrong TYPES (bool/float/str would int-coerce silently); the default env
        # build inspects ONLY the cash symbol (no spurious warnings for unknown non-cash tickers -- the
        # afdf0a6 regression); the shared transition-cost helper rejects non-finite/negative cost scalars; and
        # env.step rejects non-integer action tensors (.long() would silently truncate a float).
        import dataclasses
        import warnings

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import (
            MinuteToHourEnvConfig,
            VectorizedMinuteToHourEnv,
            transition_trade_cost_bps,
            validate_cash_index_for_actions,
        )
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        device = torch.device("cpu")

        def split(action_names):
            n = len(action_names)
            return HourFromMinuteDataSplit(
                name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
                next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=list(action_names),
                minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, n)),
                action_valid_mask=torch.ones((2, n), dtype=torch.bool), label_valid_mask=torch.ones((2, n), dtype=torch.bool),
                valid_start_indices=torch.tensor([0, 1]), valid_index_mask=torch.tensor([True, True]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        base = default_minute_to_hour_constraints()

        # (1) The validator: a valid int 0 -> "CASH" returns 0; wrong types and bad indices fail closed
        # (bool/float/str must NOT silently int-coerce; out-of-range and in-range-but-not-cash both raise).
        self.assertEqual(validate_cash_index_for_actions(["CASH", "QQQ"], 0), 0)
        for bad in (True, 1.0, "0", None):
            with self.assertRaises(ValueError):
                validate_cash_index_for_actions(["CASH", "QQQ"], bad)
        with self.assertRaises(ValueError):
            validate_cash_index_for_actions(["CASH", "QQQ"], 5)
        with self.assertRaises(ValueError):
            validate_cash_index_for_actions(["CASH", "QQQ"], 1)

        # (2) Default (shadow-off) env build with an UNKNOWN non-cash ticker emits NO warning: only the cash
        # symbol is inspected, not every action (all-action metadata build is gated behind the shadow flag).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VectorizedMinuteToHourEnv(
                split(["CASH", "UNSEEN_TICKER_ZZ"]),
                MinuteToHourEnvConfig(num_envs=1, episode_length=5, constraints=base), device)
        self.assertEqual(list(caught), [])

        # (3) The evaluator fails closed on a non-cash cash_index too (the env-vs-eval consistency this fixes).
        with self.assertRaises(ValueError):
            evaluate_minute_to_hour_policy(
                split(["CASH", "QQQ"]), _ConstantActionModel(2, 0), device=device,
                constraints=dataclasses.replace(base, cash_index=1))

        # (4) The shared transition-cost helper rejects non-finite / negative cost scalars (a NaN/negative
        # one_way_cost or extra_switch would otherwise produce a garbage but silent cost ledger).
        prev = torch.tensor([0], dtype=torch.long)
        act = torch.tensor([1], dtype=torch.long)
        with self.assertRaises(ValueError):
            transition_trade_cost_bps(prev, act, constraints=dataclasses.replace(base, one_way_cost_bps=-1.0),
                                      cash_idle_penalty_bps=0.0)
        with self.assertRaises(ValueError):
            transition_trade_cost_bps(prev, act, constraints=dataclasses.replace(base, extra_switch_penalty_bps=float("inf")),
                                      cash_idle_penalty_bps=0.0)

        # (5) env.step fails closed on a non-integer action tensor (float would truncate to the wrong index).
        env = VectorizedMinuteToHourEnv(
            split(["CASH", "QQQ"]), MinuteToHourEnvConfig(num_envs=1, episode_length=5, constraints=base), device)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(torch.tensor([0.0]))

    def _two_action_split(self, action_names):
        n = len(action_names)
        return HourFromMinuteDataSplit(
            name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
            next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=list(action_names),
            minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, n)),
            action_valid_mask=torch.ones((2, n), dtype=torch.bool), label_valid_mask=torch.ones((2, n), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1]), valid_index_mask=torch.tensor([True, True]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )

    def test_minute_to_hour_initial_action_validation(self) -> None:
        # Review #6 follow-up: initial_action gets the SAME strict discipline as cash_index in BOTH the env and
        # the evaluator -- int(True)/int(0.9)/int("0") would otherwise silently start the rollout from the
        # wrong action. A valid int start is accepted; bool/float/string/out-of-range fail closed.
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        device = torch.device("cpu")
        base = default_minute_to_hour_constraints()
        VectorizedMinuteToHourEnv(
            self._two_action_split(["CASH", "QQQ"]),
            MinuteToHourEnvConfig(num_envs=1, episode_length=5, initial_action=1, constraints=base), device)
        for bad in (True, 0.9, "0", 5):
            with self.assertRaises(ValueError):
                VectorizedMinuteToHourEnv(
                    self._two_action_split(["CASH", "QQQ"]),
                    MinuteToHourEnvConfig(num_envs=1, episode_length=5, initial_action=bad, constraints=base), device)
            with self.assertRaises(ValueError):
                evaluate_minute_to_hour_policy(
                    self._two_action_split(["CASH", "QQQ"]), _ConstantActionModel(2, 0), device=device,
                    initial_action=bad)

    def test_minute_to_hour_step_action_tensor_validation(self) -> None:
        # Review #6 follow-up: the env boundary fails closed on a malformed action tensor BEFORE gather --
        # wrong dtype (float/bool), wrong shape/rank, out-of-range / negative index (and wrong device when
        # CUDA is present). A valid (num_envs,) integer tensor passes through, returned as long.
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv

        device = torch.device("cpu")
        env = VectorizedMinuteToHourEnv(
            self._two_action_split(["CASH", "QQQ"]),
            MinuteToHourEnvConfig(num_envs=2, episode_length=5, constraints=default_minute_to_hour_constraints()),
            device)
        ok = env._validate_step_actions(torch.tensor([0, 1], dtype=torch.int32))
        self.assertEqual(ok.dtype, torch.long)
        for bad in (
            torch.tensor([0.0, 1.0]),        # float dtype (would truncate)
            torch.tensor([True, False]),     # bool dtype
            torch.tensor([0]),               # wrong shape (1,) != (2,)
            torch.tensor(0, dtype=torch.int32),  # 0-d scalar (shape () != (2,))
            torch.tensor([[0], [1]]),        # wrong rank (2, 1)
            torch.tensor([0, 5]),            # out of range (>= action_count)
            torch.tensor([-1, 0]),           # negative
        ):
            with self.assertRaises(ValueError):
                env._validate_step_actions(bad)
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                env._validate_step_actions(torch.tensor([0, 1], device="cuda"))

    def test_transition_trade_cost_bps_rejects_bad_scalars(self) -> None:
        # Review #6 follow-up: the shared cost ledger rejects bool / NaN / inf / negative / non-numeric-string
        # bps scalars (via execution._coerce_finite_nonnegative) and bool/float/string cash_index, so a direct
        # caller (not just env/eval, which validate upstream) cannot produce a silently-garbage cost.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import transition_trade_cost_bps

        base = default_minute_to_hour_constraints()
        prev = torch.tensor([0], dtype=torch.long)
        act = torch.tensor([1], dtype=torch.long)
        for bad in (True, float("nan"), float("inf"), -1.0, "x"):
            with self.assertRaises(ValueError):
                transition_trade_cost_bps(prev, act, constraints=base, cash_idle_penalty_bps=bad)
        for field in ("one_way_cost_bps", "extra_switch_penalty_bps"):
            for bad in (True, float("inf"), -1.0, "x"):
                with self.assertRaises(ValueError):
                    transition_trade_cost_bps(prev, act, constraints=dataclasses.replace(base, **{field: bad}),
                                              cash_idle_penalty_bps=0.0)
        for bad in (True, 0.9, "0"):
            with self.assertRaises(ValueError):
                transition_trade_cost_bps(prev, act, constraints=dataclasses.replace(base, cash_index=bad),
                                          cash_idle_penalty_bps=0.0)
        # A numeric STRING is intentionally accepted and parsed (execution-module config contract); only bool /
        # NaN / inf / negative / non-numeric strings fail closed. CASH(0)->QQQ(1) is one leg, so "2.0" -> 2.0bps.
        ok = transition_trade_cost_bps(prev, act, constraints=dataclasses.replace(base, one_way_cost_bps="2.0"),
                                       cash_idle_penalty_bps=0.0)
        self.assertEqual(float(ok.leg_cost_bps[0].item()), 2.0)

    def test_minute_to_hour_baselines_rejects_non_cash_cash_index(self) -> None:
        # Review #6 follow-up: evaluate_minute_to_hour_baselines reads cash_index (to skip the cash baseline /
        # set initial_action) before delegating, so it must fail closed on a non-cash index too.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.training.minute_to_hour import evaluate_minute_to_hour_baselines

        base = default_minute_to_hour_constraints()
        with self.assertRaises(ValueError):
            evaluate_minute_to_hour_baselines(
                self._two_action_split(["CASH", "QQQ"]), device=torch.device("cpu"),
                constraints=dataclasses.replace(base, cash_index=1))

    def test_minute_to_hour_env_device_normalization(self) -> None:
        # The env pins self.device to a CONCRETE ordinal: resolve_torch_device returns an ordinal-free
        # torch.device("cuda"), but tensors allocated on it report cuda:<idx>, so _validate_step_actions would
        # reject valid CUDA actions if self.device stayed ordinal-free. CPU is a no-op; CUDA is the real test.
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv

        cfg = MinuteToHourEnvConfig(num_envs=2, episode_length=5, constraints=default_minute_to_hour_constraints())
        env = VectorizedMinuteToHourEnv(self._two_action_split(["CASH", "QQQ"]), cfg, torch.device("cpu"))
        self.assertEqual(env.device.type, "cpu")
        env._validate_step_actions(torch.zeros(2, dtype=torch.long, device=env.device))  # no raise
        if torch.cuda.is_available():
            env_cuda = VectorizedMinuteToHourEnv(
                self._two_action_split(["CASH", "QQQ"]), cfg, torch.device("cuda"))
            self.assertIsNotNone(env_cuda.device.index)  # concrete ordinal, e.g. cuda:0
            # A tensor built with ordinal-free "cuda" reports cuda:<current> and MUST be accepted.
            env_cuda._validate_step_actions(torch.zeros(2, dtype=torch.long, device=torch.device("cuda")))

    def test_concrete_torch_device_and_env_device_matches_tensors(self) -> None:
        # concrete_torch_device pins ordinal-free CUDA to cuda:<current>; CPU/explicit-ordinal pass through.
        # The env derives self.device from the moved tensor, so it equals what indexed tensors report.
        from rl_quant.core import concrete_torch_device
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv

        self.assertEqual(concrete_torch_device("cpu"), torch.device("cpu"))
        self.assertEqual(concrete_torch_device(torch.device("cpu")), torch.device("cpu"))
        cfg = MinuteToHourEnvConfig(num_envs=2, episode_length=5, constraints=default_minute_to_hour_constraints())
        env = VectorizedMinuteToHourEnv(self._two_action_split(["CASH", "QQQ"]), cfg, torch.device("cpu"))
        self.assertEqual(env.device, env.data.minute_features.device)
        if torch.cuda.is_available():
            dev = concrete_torch_device("cuda")
            self.assertEqual(dev.type, "cuda")
            self.assertIsNotNone(dev.index)
            env_cuda = VectorizedMinuteToHourEnv(
                self._two_action_split(["CASH", "QQQ"]), cfg, torch.device("cuda"))
            self.assertIsNotNone(env_cuda.device.index)
            self.assertEqual(env_cuda.device, env_cuda.data.minute_features.device)

    def test_minute_to_hour_env_entry_constraint_and_flag_validation(self) -> None:
        # Entry-point validation: a non-bool count_etf (truthy "false" would skew the action mask) and a
        # non-bool governed flag (execution_env_reward_shadow="false" would silently ENABLE the shadow) must
        # both fail closed at env CONSTRUCTION, before any mask/observe can consume the malformed value.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import (
            MinuteToHourEnvConfig,
            VectorizedMinuteToHourEnv,
            validate_minute_to_hour_constraints,
        )

        device = torch.device("cpu")
        base = default_minute_to_hour_constraints()
        with self.assertRaises(ValueError):
            VectorizedMinuteToHourEnv(
                self._two_action_split(["CASH", "QQQ"]),
                MinuteToHourEnvConfig(num_envs=1, episode_length=5,
                                      constraints=dataclasses.replace(base, count_etf_to_etf_as_two_legs="false")),
                device)
        with self.assertRaises(ValueError):
            VectorizedMinuteToHourEnv(
                self._two_action_split(["CASH", "QQQ"]),
                MinuteToHourEnvConfig(num_envs=1, episode_length=5, execution_env_reward_shadow="false",
                                      constraints=base),
                device)
        # The shared constraint validator returns the NORMALIZED constraints (cash_index field) and rejects a
        # non-cash cash_index.
        self.assertEqual(validate_minute_to_hour_constraints(base, ["CASH", "QQQ"]).cash_index, 0)
        with self.assertRaises(ValueError):
            validate_minute_to_hour_constraints(dataclasses.replace(base, cash_index=1), ["CASH", "QQQ"])

    def test_train_minute_to_hour_rejects_non_bool_feature_flags(self) -> None:
        # Governed model-input flags must be real bools at train entry (a truthy string would flip the model
        # contract / replay schema). The check is at the top of train_minute_to_hour_dqn, before any work.
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit, MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        def make_split(name: str) -> HourFromMinuteDataSplit:
            return HourFromMinuteDataSplit(
                name=name, decision_timestamps=["2026-01-02T14:30:00+00:00", "2026-01-03T14:30:00+00:00"],
                next_timestamps=["2026-01-02T15:30:00+00:00", "2026-01-03T15:30:00+00:00"],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, 2)),
                action_valid_mask=torch.ones((2, 2), dtype=torch.bool), label_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                valid_start_indices=torch.tensor([0]), valid_index_mask=torch.tensor([True, False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        learning = DQNLearningConfig(
            num_envs=1, episode_length=2, replay_capacity=8, batch_size=2, train_steps=1, warmup_steps=0,
            gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=1, epsilon_start=0.0,
            epsilon_end=0.0, eval_interval=1, grad_clip=1.0,
        )
        for flag in ("use_transition_features", "use_dynamic_transition_features"):
            config = MinuteToHourTrainingConfig(
                env=MinuteToHourEnvConfig(num_envs=1, episode_length=2), learning=learning,
                d_model=8, n_heads=1, minute_layers=1, hour_layers=1, feedforward_dim=8, action_embedding_dim=2,
                **{flag: "false"},
            )
            with self.assertRaises(ValueError):
                train_minute_to_hour_dqn(make_split("train"), make_split("val"), device=torch.device("cpu"), config=config)

    def test_minute_to_hour_reward_scale_validation(self) -> None:
        # reward_scale multiplies every reward and normalises the shadow bps artifact; a zero / negative /
        # non-finite / bool value must fail closed in BOTH the env and the evaluator.
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        base = default_minute_to_hour_constraints()
        device = torch.device("cpu")
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                VectorizedMinuteToHourEnv(
                    self._two_action_split(["CASH", "QQQ"]),
                    MinuteToHourEnvConfig(num_envs=1, episode_length=5, reward_scale=bad, constraints=base), device)
            with self.assertRaises(ValueError):
                evaluate_minute_to_hour_policy(
                    self._two_action_split(["CASH", "QQQ"]), _ConstantActionModel(2, 0), device=device,
                    reward_scale=bad)

    def test_transition_trade_cost_bps_action_count_range_and_count_etf_bool(self) -> None:
        # action_count (optional) range-checks the action indices for a DIRECT caller; count_etf must be a real
        # bool (a truthy string would silently pick the two-leg path).
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import transition_trade_cost_bps

        base = default_minute_to_hour_constraints()
        long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
        # in-range with action_count=2 is fine; out-of-range / negative fail closed.
        transition_trade_cost_bps(long([0]), long([1]), constraints=base, cash_idle_penalty_bps=0.0, action_count=2)
        for prev, act in [([-1], [0]), ([0], [-1]), ([0], [2]), ([2], [0])]:
            with self.assertRaises(ValueError):
                transition_trade_cost_bps(long(prev), long(act), constraints=base, cash_idle_penalty_bps=0.0,
                                          action_count=2)
        # Without action_count the range is NOT checked (backward compatible) -- an OOR index does not raise here.
        transition_trade_cost_bps(long([0]), long([5]), constraints=base, cash_idle_penalty_bps=0.0)
        # count_etf_to_etf_as_two_legs must be a real bool, not a truthy/other value.
        for bad in ("false", 1, 0, None):
            with self.assertRaises(ValueError):
                transition_trade_cost_bps(long([0]), long([1]),
                                          constraints=dataclasses.replace(base, count_etf_to_etf_as_two_legs=bad),
                                          cash_idle_penalty_bps=0.0)
        # action_count, when supplied, must itself be a positive integer (reject bool/float/<=0)...
        for bad_ac in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                transition_trade_cost_bps(long([0]), long([1]), constraints=base, cash_idle_penalty_bps=0.0,
                                          action_count=bad_ac)
        # ...and cash_index must be inside it (a direct caller with cash_index=99 + action_count=2 fails closed).
        with self.assertRaises(ValueError):
            transition_trade_cost_bps(long([0]), long([1]), constraints=dataclasses.replace(base, cash_index=99),
                                      cash_idle_penalty_bps=0.0, action_count=2)

    def test_minute_to_hour_shadow_artifact_flags_incomplete_metadata(self) -> None:
        # PR-3 auditability: with shadow ON and an UNKNOWN (un-metadata'd) non-cash ticker, the artifact must
        # flag execution_shadow_action_metadata_complete=False and list the unknown symbol, so PR-4 can fail
        # closed on it (an unknown leveraged/inverse instrument would otherwise be priced as 1x long).
        import warnings

        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit, MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        def make_split(name: str, dates: list[str]) -> HourFromMinuteDataSplit:
            n = len(dates)
            return HourFromMinuteDataSplit(
                name=name, decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
                next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "UNSEEN_TICKER_ZZ"],
                minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((n, 1, 1)), action_returns=torch.zeros((n, 2)),
                action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                valid_start_indices=torch.arange(n - 1, dtype=torch.long), valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        train = make_split("train", ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-02", "2026-05-20"])
        val = make_split("val", ["2026-06-01", "2026-06-02"])
        learning = DQNLearningConfig(
            num_envs=2, episode_length=3, replay_capacity=64, batch_size=4, train_steps=6, warmup_steps=2,
            gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=3, epsilon_start=0.2,
            epsilon_end=0.0, eval_interval=4, grad_clip=1.0,
        )
        config = MinuteToHourTrainingConfig(
            env=MinuteToHourEnvConfig(num_envs=2, episode_length=3, execution_env_reward_shadow=True),
            learning=learning, d_model=16, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16,
            action_embedding_dim=4,
        )
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unknown-symbol metadata fallback warns by design
            artifacts = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config)[1]
        self.assertFalse(artifacts["execution_shadow_action_metadata_complete"])
        self.assertIn("UNSEEN_TICKER_ZZ", artifacts["execution_shadow_unknown_action_symbols"])
        self.assertEqual(artifacts["execution_shadow_weight_semantics_status"], "unresolved")

    def test_minute_to_hour_eval_episode_length_and_cash_idle_validation(self) -> None:
        # The evaluator now fails closed on a bad episode_length (the bare int(episode_length or ...) silently
        # turned 0 -> default, True -> 1, 1.9 -> 1) and on a bad cash_idle_penalty_bps, matching the env. None
        # episode_length still means "use the full valid-start span".
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        device = torch.device("cpu")
        names = ["CASH", "QQQ"]
        evaluate_minute_to_hour_policy(self._two_action_split(names), _ConstantActionModel(2, 0),
                                       device=device, episode_length=None)
        for bad in (0, -1, True, 1.9):
            with self.assertRaises(ValueError):
                evaluate_minute_to_hour_policy(self._two_action_split(names), _ConstantActionModel(2, 0),
                                               device=device, episode_length=bad)
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                evaluate_minute_to_hour_policy(self._two_action_split(names), _ConstantActionModel(2, 0),
                                               device=device, cash_idle_penalty_bps=bad)

    def test_minute_to_hour_env_step_treats_nonfinite_label_as_unusable(self) -> None:
        # Env/eval parity: a label the mask calls "valid" but whose return is NaN/inf must NOT be traded on
        # (the env would otherwise produce a NaN reward). The env falls back to CASH exactly like the evaluator.
        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv

        device = torch.device("cpu")
        # Row 0: QQQ is mask-valid but its return is NaN; CASH(0) is finite. Requesting QQQ must execute CASH.
        split = HourFromMinuteDataSplit(
            name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
            next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((2, 1, 1)),
            action_returns=torch.tensor([[0.0, float("nan")], [0.0, 0.0]]),
            action_valid_mask=torch.ones((2, 2), dtype=torch.bool), label_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0]), valid_index_mask=torch.tensor([True, False]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        env = VectorizedMinuteToHourEnv(
            split, MinuteToHourEnvConfig(num_envs=1, episode_length=5, constraints=default_minute_to_hour_constraints()),
            device)
        env.reset()
        out = env.step(torch.tensor([1]))  # request QQQ (mask-valid but NaN return) -> must fall back to CASH(0)
        self.assertEqual(int(out["actions"][0].item()), 0)
        self.assertTrue(bool(torch.isfinite(out["rewards"][0]).item()))

    def test_minute_to_hour_eval_rejects_empty_valid_start_split(self) -> None:
        # An evaluation split with no valid decision rows fails closed (a zero/degenerate metric would
        # otherwise look like a legitimate result), mirroring the env's start-index-pool guard.
        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        empty = HourFromMinuteDataSplit(
            name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
            next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, 2)),
            action_valid_mask=torch.ones((2, 2), dtype=torch.bool), label_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([], dtype=torch.long), valid_index_mask=torch.tensor([False, False]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        with self.assertRaises(ValueError):
            evaluate_minute_to_hour_policy(empty, _ConstantActionModel(2, 0), device=torch.device("cpu"),
                                           constraints=default_minute_to_hour_constraints())

    def test_minute_to_hour_env_eval_golden_ledger_parity(self) -> None:
        # GOLDEN PARITY: the vectorized env (step) and the sequential evaluator must compute the SAME ledger for
        # the same requested-action sequence over the same rows -- executed action (incl. the missing-label ->
        # CASH fallback applied INDEPENDENTLY by each path), order legs, and reward (= reward_scale * eval
        # net_return), cumulatively too. This is the guardrail against future drift between the two reward
        # paths: they share transition_trade_cost_bps but reconstruct the rest of the rollout separately.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv
        from rl_quant.training.minute_to_hour import evaluate_minute_to_hour_policy

        class FavorQQQ(nn.Module):  # always strongly prefers QQQ(1): enters from CASH, then holds
            def forward(self, minute, mask, hour, previous_actions, constraint_features,
                        action_features=None, dynamic_state=None):
                q = torch.zeros((previous_actions.shape[0], 2), device=previous_actions.device)
                q[:, 1] = 1000.0
                return q

        device = torch.device("cpu")
        n = 6  # rows 0..4 are decisions; row 5 provides the next-state boundary
        # QQQ return is NaN at row 3 -> both paths must de-risk to CASH there; CASH return is 0 everywhere. All
        # timestamps share one day (no day reset); rows are contiguous (no segment reset after the first).
        qqq = [0.01, -0.004, 0.02, float("nan"), 0.013, 0.0]
        base = HourFromMinuteDataSplit(
            name="t",
            decision_timestamps=[f"2026-01-02T1{i}:30:00+00:00" for i in range(n)],
            next_timestamps=[f"2026-01-02T1{i + 1}:30:00+00:00" for i in range(n)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1)),
            action_returns=torch.tensor([[0.0, q] for q in qqq]),
            action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3, 4]), valid_index_mask=torch.ones(n, dtype=torch.bool),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        cons = dataclasses.replace(
            default_minute_to_hour_constraints(), max_switches_per_day=None, max_switches_per_episode=None,
            q_switch_margin_bps=0.0, one_way_cost_bps=2.0, extra_switch_penalty_bps=3.0, min_hold_bars=1, cooldown_bars=0)
        reward_scale, cash_idle = 10_000.0, 5.0

        eval_result = evaluate_minute_to_hour_policy(
            base, FavorQQQ(), device=device, initial_action=0, constraints=cons,
            episode_length=10, reward_scale=reward_scale, cash_idle_penalty_bps=cash_idle, capture_rollout=True)
        records = eval_result.rollout_records
        self.assertEqual(len(records), 5)
        # Non-trivial path: an entering switch, two holds, the NaN-label de-risk to CASH, and a re-entry.
        self.assertEqual([int(r["executed_action"]) for r in records], [1, 1, 1, 0, 1])

        # Drive the env over the same rows from a single deterministic start (0), feeding the eval's REQUESTED
        # actions so the env runs its OWN mask + finite-label fallback independently.
        env = VectorizedMinuteToHourEnv(
            dataclasses.replace(base, valid_start_indices=torch.tensor([0])),
            MinuteToHourEnvConfig(num_envs=1, episode_length=10, reward_scale=reward_scale,
                                  initial_action=0, cash_idle_penalty_bps=cash_idle, constraints=cons),
            device)
        env.reset()
        env_equity, env_switches, env_legs_total, prev = 1.0, 0, 0.0, 0
        for rec in records:
            out = env.step(torch.tensor([int(rec["requested_action"])]))
            executed = int(out["actions"][0].item())
            self.assertEqual(executed, int(rec["executed_action"]))                                   # fallback parity
            self.assertAlmostEqual(float(out["legs"][0].item()), float(rec["market_order_legs"]), places=6)  # legs
            net = float(out["rewards"][0].item()) / reward_scale
            self.assertAlmostEqual(net, float(rec["net_return"]), places=6)                           # reward/cost
            env_equity *= 1.0 + net
            env_switches += int(executed != prev)
            env_legs_total += float(out["legs"][0].item())
            prev = executed

        # Cumulative parity.
        self.assertAlmostEqual(env_equity - 1.0, eval_result.total_return, places=6)
        self.assertEqual(env_switches, eval_result.allocation_switches)
        self.assertAlmostEqual(env_legs_total, eval_result.market_order_legs, places=6)

    def test_minute_to_hour_env_eval_dynamic_state_parity(self) -> None:
        # GOLDEN PARITY (dynamic state): the held-position excursion (unrealized_pnl / MAE / MFE) the evaluator
        # feeds its network must equal the env's position_dynamic entering the SAME decision -- this is the
        # off-by-one-prone "state before action" surface. Both advance via the shared advance_position_excursion
        # on the executed action's gross return with held = not is_switch; this pins them step-for-step.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv
        from rl_quant.trading_constraints import DYNAMIC_TRANSITION_FEATURE_DIM
        from rl_quant.training.minute_to_hour import evaluate_minute_to_hour_policy

        class DynamicSpy(nn.Module):  # dynamic-aware so the eval tracks + feeds dynamic state; records what it sees
            dynamic_feature_dim = DYNAMIC_TRANSITION_FEATURE_DIM

            def __init__(self) -> None:
                super().__init__()
                self.seen: list[torch.Tensor] = []

            def forward(self, minute, mask, hour, previous_actions, constraint_features,
                        action_features=None, dynamic_state=None):
                self.seen.append(dynamic_state.detach().clone())
                q = torch.zeros((previous_actions.shape[0], 2), device=previous_actions.device)
                q[:, 1] = 1000.0  # same fixed preference as the ledger-parity test (deterministic sequence)
                return q

        device = torch.device("cpu")
        n = 6
        qqq = [0.01, -0.004, 0.02, float("nan"), 0.013, 0.0]
        base = HourFromMinuteDataSplit(
            name="t",
            decision_timestamps=[f"2026-01-02T1{i}:30:00+00:00" for i in range(n)],
            next_timestamps=[f"2026-01-02T1{i + 1}:30:00+00:00" for i in range(n)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1)),
            action_returns=torch.tensor([[0.0, q] for q in qqq]),
            action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3, 4]), valid_index_mask=torch.ones(n, dtype=torch.bool),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        cons = dataclasses.replace(
            default_minute_to_hour_constraints(), max_switches_per_day=None, max_switches_per_episode=None,
            q_switch_margin_bps=0.0, one_way_cost_bps=2.0, extra_switch_penalty_bps=3.0, min_hold_bars=1, cooldown_bars=0)

        spy = DynamicSpy()
        eval_result = evaluate_minute_to_hour_policy(
            base, spy, device=device, initial_action=0, constraints=cons,
            episode_length=10, reward_scale=10_000.0, cash_idle_penalty_bps=5.0, capture_rollout=True)
        records = eval_result.rollout_records
        self.assertEqual(len(spy.seen), 5)  # one dynamic-state observation per decision row

        env = VectorizedMinuteToHourEnv(
            dataclasses.replace(base, valid_start_indices=torch.tensor([0])),
            MinuteToHourEnvConfig(num_envs=1, episode_length=10, reward_scale=10_000.0,
                                  initial_action=0, cash_idle_penalty_bps=5.0, constraints=cons),
            device)
        env.reset()
        for step, rec in enumerate(records):
            out = env.step(torch.tensor([int(rec["requested_action"])]))
            # The env's dynamic features ENTERING this step must equal what the evaluator fed its network here.
            self.assertTrue(torch.allclose(out["position_dynamic"], spy.seen[step], atol=1e-6))
        # Sanity: the excursion was actually non-trivial (not all zeros) at least once after the first hold.
        self.assertTrue(any(bool(s.abs().sum().item() > 0.0) for s in spy.seen))

    def test_minute_to_hour_env_eval_day_boundary_cap_parity(self) -> None:
        # GOLDEN PARITY (day boundary + turnover cap): with max_switches_per_day=1, a day-1 entry exhausts the
        # daily cap; the cap must RESET on the day-2 boundary so a day-2 re-entry is selectable again. The env
        # and the evaluator must reset the per-day counter at the same point AND the env's OWN mask must agree
        # with the evaluator's (so feeding the env the requested QQQ re-enters on day 2 instead of mask-falling
        # back to CASH). A wrong env day-reset would mask the day-2 entry and diverge here.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv
        from rl_quant.training.minute_to_hour import evaluate_minute_to_hour_policy

        class FavorQQQ(nn.Module):
            def forward(self, minute, mask, hour, previous_actions, constraint_features,
                        action_features=None, dynamic_state=None):
                q = torch.zeros((previous_actions.shape[0], 2), device=previous_actions.device)
                q[:, 1] = 1000.0
                return q

        device = torch.device("cpu")
        # rows 0,1 -> day 1; rows 2,3 -> day 2 (row 4 = next-state boundary). QQQ is NaN at row 1 -> day-1
        # de-risk to CASH, so the position entering day 2 is CASH and a fresh QQQ entry is needed there.
        dates = ["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03", "2026-01-03"]
        qqq = [0.01, float("nan"), 0.02, 0.015, 0.0]
        n = 5
        base = HourFromMinuteDataSplit(
            name="t",
            decision_timestamps=[f"{d}T1{i}:30:00+00:00" for i, d in enumerate(dates)],
            next_timestamps=[f"{d}T1{i + 1}:30:00+00:00" for i, d in enumerate(dates)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1)),
            action_returns=torch.tensor([[0.0, q] for q in qqq]),
            action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            valid_start_indices=torch.tensor([0, 1, 2, 3]), valid_index_mask=torch.ones(n, dtype=torch.bool),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        cons = dataclasses.replace(
            default_minute_to_hour_constraints(), max_switches_per_day=1, max_switches_per_episode=None,
            q_switch_margin_bps=0.0, one_way_cost_bps=2.0, extra_switch_penalty_bps=3.0, min_hold_bars=1, cooldown_bars=0)

        eval_result = evaluate_minute_to_hour_policy(
            base, FavorQQQ(), device=device, initial_action=0, constraints=cons,
            episode_length=10, reward_scale=10_000.0, cash_idle_penalty_bps=5.0, capture_rollout=True)
        records = eval_result.rollout_records
        # Day 1: enter QQQ, then NaN-label de-risk to CASH (cap now exhausted on day 1). Day 2: cap reset lets
        # the QQQ entry happen again, then hold. If the cap did NOT reset, row 2 would mask QQQ -> CASH.
        self.assertEqual([int(r["executed_action"]) for r in records], [1, 0, 1, 1])

        env = VectorizedMinuteToHourEnv(
            dataclasses.replace(base, valid_start_indices=torch.tensor([0])),
            MinuteToHourEnvConfig(num_envs=1, episode_length=10, reward_scale=10_000.0,
                                  initial_action=0, cash_idle_penalty_bps=5.0, constraints=cons),
            device)
        env.reset()
        for rec in records:
            out = env.step(torch.tensor([int(rec["requested_action"])]))
            self.assertEqual(int(out["actions"][0].item()), int(rec["executed_action"]))       # mask/cap/day-reset parity
            self.assertAlmostEqual(float(out["legs"][0].item()), float(rec["market_order_legs"]), places=6)
            self.assertAlmostEqual(float(out["rewards"][0].item()) / 10_000.0, float(rec["net_return"]), places=6)

    def test_minute_to_hour_constraints_normalized_object(self) -> None:
        # validate_minute_to_hour_constraints returns a FROZEN NormalizedMinuteToHourConstraints whose fields
        # are canonical Python types (a numeric-string bps -> float, a numpy int index/count -> int), and the
        # env stores + uses that object -- so a non-canonical config value can never reach the runtime raw.
        import dataclasses

        import numpy as np

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import (
            MinuteToHourEnvConfig,
            NormalizedMinuteToHourConstraints,
            VectorizedMinuteToHourEnv,
            validate_minute_to_hour_constraints,
        )

        names = ["CASH", "QQQ"]
        # Non-canonical-but-valid inputs: a numeric-string bps and numpy integer index/bar counts.
        raw = dataclasses.replace(
            default_minute_to_hour_constraints(),
            one_way_cost_bps="2.0", cash_index=np.int64(0), min_hold_bars=np.int64(1))
        norm = validate_minute_to_hour_constraints(raw, names)
        self.assertIsInstance(norm, NormalizedMinuteToHourConstraints)
        self.assertIsInstance(norm.one_way_cost_bps, float)
        self.assertEqual(norm.one_way_cost_bps, 2.0)
        self.assertIsInstance(norm.cash_index, int)
        self.assertNotIsInstance(norm.cash_index, np.integer)
        self.assertIsInstance(norm.min_hold_bars, int)
        with self.assertRaises(dataclasses.FrozenInstanceError):  # frozen: validated values can't be mutated
            norm.cash_index = 1  # type: ignore[misc]
        # The env stores and exposes the normalized object.
        env = VectorizedMinuteToHourEnv(
            self._two_action_split(names),
            MinuteToHourEnvConfig(num_envs=1, episode_length=5, constraints=raw), torch.device("cpu"))
        self.assertIsInstance(env.constraints, NormalizedMinuteToHourConstraints)
        self.assertEqual(env.constraints.one_way_cost_bps, 2.0)
        self.assertIsInstance(env.constraints.one_way_cost_bps, float)

    def test_minute_to_hour_requires_usable_cash_on_valid_rows(self) -> None:
        # CASH is the forced safety fallback; it must be USABLE (label-valid AND finite return) on every valid
        # decision row. BOTH a non-finite CASH return AND a finite-but-label-invalid CASH must fail closed at
        # env construction AND at evaluator entry (the fallback reads the CASH return directly).
        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv
        from rl_quant.training.minute_to_hour import _ConstantActionModel, evaluate_minute_to_hour_policy

        device = torch.device("cpu")
        base = default_minute_to_hour_constraints()

        def split(action_returns, label_valid_mask) -> HourFromMinuteDataSplit:
            return HourFromMinuteDataSplit(
                name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
                next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((2, 1, 1)), action_returns=action_returns,
                action_valid_mask=torch.ones((2, 2), dtype=torch.bool), label_valid_mask=label_valid_mask,
                valid_start_indices=torch.tensor([0]), valid_index_mask=torch.tensor([True, False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1)

        nan_cash = split(torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]), torch.ones((2, 2), dtype=torch.bool))
        # CASH return finite (0.0) but its label is marked invalid at the valid row 0 -> still unusable.
        label_invalid_cash = split(torch.zeros((2, 2)), torch.tensor([[False, True], [True, True]]))
        for bad in (nan_cash, label_invalid_cash):
            with self.assertRaises(ValueError):
                VectorizedMinuteToHourEnv(bad, MinuteToHourEnvConfig(num_envs=1, episode_length=5, constraints=base), device)
            with self.assertRaises(ValueError):
                evaluate_minute_to_hour_policy(bad, _ConstantActionModel(2, 0), device=device, constraints=base)

    def _shadow_resume_split(self, name: str, dates: list[str]):
        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit
        n = len(dates)
        return HourFromMinuteDataSplit(
            name=name, decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
            next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1)), action_returns=torch.zeros((n, 2)),
            action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            valid_start_indices=torch.arange(n - 1, dtype=torch.long), valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1)

    def test_minute_to_hour_shadow_aggregates_are_resume_safe(self) -> None:
        # The PR-3 shadow aggregates are a running sum + count saved in the checkpoint, so a resumed run's
        # artifact covers the WHOLE run. After training 3 steps then resuming to 5, the checkpoint's
        # shadow_delta_count must be 5 (full), not 2 (post-resume only) -- the bug this fixes.
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        train = self._shadow_resume_split("train", ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
        val = self._shadow_resume_split("val", ["2026-02-01", "2026-02-02"])

        def cfg(train_steps: int, state_path, *, resume: bool) -> MinuteToHourTrainingConfig:
            return MinuteToHourTrainingConfig(
                env=MinuteToHourEnvConfig(num_envs=2, episode_length=2, execution_env_reward_shadow=True),
                learning=DQNLearningConfig(
                    num_envs=2, episode_length=2, replay_capacity=16, batch_size=2, train_steps=train_steps,
                    warmup_steps=1, gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=2,
                    epsilon_start=0.1, epsilon_end=0.0, eval_interval=2, grad_clip=1.0, use_amp=False),
                d_model=16, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16, action_embedding_dim=4,
                resume_training_state=state_path if resume else None,
                checkpoint_training_state=state_path, checkpoint_every_steps=1)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.pt"
            torch.manual_seed(123)
            _, first = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=cfg(3, state_path, resume=False))
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["shadow_delta_count"], 3)  # 3 steps accumulated pre-resume
            self.assertIsNotNone(first["execution_shadow_reward_delta_mean"])

            _, resumed = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=cfg(5, state_path, resume=True))
            saved_again = torch.load(state_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved_again["shadow_delta_count"], 5)  # full run (3 restored + 2 more), not just 2
            self.assertIsNotNone(resumed["execution_shadow_reward_delta_mean"])

    def test_minute_to_hour_resume_rejects_changed_economics(self) -> None:
        # A checkpoint must not be resumed with DIFFERENT economics (here: a changed one_way_cost_bps). The run
        # semantics hash saved in the checkpoint mismatches the resuming run's, so resume fails closed -- same
        # tensor shapes are not enough to share a run.
        import dataclasses

        from rl_quant.core import DQNLearningConfig
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.minute_to_hour_transformer import (
            MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        train = self._shadow_resume_split("train", ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
        val = self._shadow_resume_split("val", ["2026-02-01", "2026-02-02"])

        def cfg(train_steps: int, state_path, *, resume: bool, constraints) -> MinuteToHourTrainingConfig:
            return MinuteToHourTrainingConfig(
                env=MinuteToHourEnvConfig(num_envs=2, episode_length=2, constraints=constraints),
                learning=DQNLearningConfig(
                    num_envs=2, episode_length=2, replay_capacity=16, batch_size=2, train_steps=train_steps,
                    warmup_steps=1, gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=2,
                    epsilon_start=0.1, epsilon_end=0.0, eval_interval=2, grad_clip=1.0, use_amp=False),
                d_model=16, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16, action_embedding_dim=4,
                resume_training_state=state_path if resume else None,
                checkpoint_training_state=state_path, checkpoint_every_steps=1)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.pt"
            torch.manual_seed(123)
            train_minute_to_hour_dqn(train, val, device=torch.device("cpu"),
                                     config=cfg(3, state_path, resume=False, constraints=default_minute_to_hour_constraints()))
            # Resume with a DIFFERENT one_way_cost_bps -> changed economics -> hash mismatch -> fail closed.
            changed = dataclasses.replace(default_minute_to_hour_constraints(), one_way_cost_bps=2.0)
            with self.assertRaises(ValueError):
                train_minute_to_hour_dqn(train, val, device=torch.device("cpu"),
                                         config=cfg(5, state_path, resume=True, constraints=changed))

    def test_minute_to_hour_pr4_execution_reward_gate(self) -> None:
        # use_execution_env_reward (PR-4) is fail-closed: it requires RESOLVED action_return_weight_semantics
        # AND complete action metadata. The flag is not a config field yet, so we set it on the env config
        # instance (a plain dataclass allows arbitrary attributes) to arm the otherwise-dormant guard.
        import dataclasses

        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        learning = DQNLearningConfig(
            num_envs=1, episode_length=2, replay_capacity=8, batch_size=2, train_steps=1, warmup_steps=0,
            gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=1, epsilon_start=0.0,
            epsilon_end=0.0, eval_interval=1, grad_clip=1.0)

        def run(train_split, val_split, *, arm: bool) -> None:
            env_cfg = MinuteToHourEnvConfig(num_envs=1, episode_length=2)
            if arm:
                env_cfg.use_execution_env_reward = True  # arm the dormant PR-4 guard (not a real field yet)
            config = MinuteToHourTrainingConfig(
                env=env_cfg, learning=learning, d_model=8, n_heads=1, minute_layers=1, hour_layers=1,
                feedforward_dim=8, action_embedding_dim=2)
            train_minute_to_hour_dqn(train_split, val_split, device=torch.device("cpu"), config=config)

        train = self._shadow_resume_split("train", ["2026-01-02", "2026-01-03"])
        val = self._shadow_resume_split("val", ["2026-02-01", "2026-02-02"])
        # (a) armed + unresolved semantics (default None) -> fail closed.
        with self.assertRaises(ValueError):
            run(train, val, arm=True)
        # (b) armed + resolved semantics BUT unknown action metadata symbol -> fail closed.
        unknown_train = dataclasses.replace(
            self._shadow_resume_split("train", ["2026-01-02", "2026-01-03"]),
            action_names=["CASH", "UNSEEN_TICKER_ZZ"],
            action_return_weight_semantics="full_capital_single_slot_returns")
        unknown_val = dataclasses.replace(
            self._shadow_resume_split("val", ["2026-02-01", "2026-02-02"]),
            action_names=["CASH", "UNSEEN_TICKER_ZZ"],
            action_return_weight_semantics="full_capital_single_slot_returns")
        with self.assertRaises(ValueError):
            run(unknown_train, unknown_val, arm=True)
        # (c) armed + a non-enum semantics string ("resolved" is not in the allowed vocabulary) -> fail closed.
        bogus_train = dataclasses.replace(
            self._shadow_resume_split("train", ["2026-01-02", "2026-01-03"]), action_return_weight_semantics="resolved")
        bogus_val = dataclasses.replace(
            self._shadow_resume_split("val", ["2026-02-01", "2026-02-02"]), action_return_weight_semantics="resolved")
        with self.assertRaises(ValueError):
            run(bogus_train, bogus_val, arm=True)
        # (d) armed + train/val disagree on semantics -> fail closed.
        mt = dataclasses.replace(
            self._shadow_resume_split("train", ["2026-01-02", "2026-01-03"]),
            action_return_weight_semantics="metadata_weighted_portfolio_returns")
        mv = dataclasses.replace(
            self._shadow_resume_split("val", ["2026-02-01", "2026-02-02"]),
            action_return_weight_semantics="full_capital_single_slot_returns")
        with self.assertRaises(ValueError):
            run(mt, mv, arm=True)
        # (e) NOT armed (the normal path) trains fine even with unresolved semantics -- the guard is dormant.
        run(train, val, arm=False)

    def test_minute_to_hour_full_constraint_and_sizing_validation(self) -> None:
        # Entry-point validation now covers the FULL constraint set that feeds masks/hysteresis/caps (not just
        # the cost-critical subset): q_switch_margin_bps (NaN would poison hysteresis), the hold/cooldown bar
        # counts, and the optional switch/order-leg caps. Plus env sizing (num_envs / episode_length positive
        # ints; cash_idle_penalty_bps finite/non-negative). Defaults pass; bad values fail closed.
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import (
            MinuteToHourEnvConfig,
            VectorizedMinuteToHourEnv,
            validate_minute_to_hour_constraints,
        )

        device = torch.device("cpu")
        names = ["CASH", "QQQ"]
        base = default_minute_to_hour_constraints()
        self.assertEqual(validate_minute_to_hour_constraints(base, names).cash_index, 0)
        bad_fields = {
            "q_switch_margin_bps": float("nan"),
            "min_hold_bars": True,
            "cooldown_bars": -1,
            "max_switches_per_day": 1.9,
            "max_switches_per_episode": -1,
            "max_order_legs_per_day": -1.0,
            "max_order_legs_per_episode": "x",
        }
        for field_name, bad in bad_fields.items():
            with self.assertRaises(ValueError):
                validate_minute_to_hour_constraints(dataclasses.replace(base, **{field_name: bad}), names)
        # None caps mean "uncapped" and are accepted.
        validate_minute_to_hour_constraints(
            dataclasses.replace(base, max_switches_per_day=None, max_order_legs_per_day=None), names)
        # Env sizing fails closed at construction.
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                VectorizedMinuteToHourEnv(self._two_action_split(names),
                                          MinuteToHourEnvConfig(num_envs=bad, episode_length=5, constraints=base), device)
            with self.assertRaises(ValueError):
                VectorizedMinuteToHourEnv(self._two_action_split(names),
                                          MinuteToHourEnvConfig(num_envs=1, episode_length=bad, constraints=base), device)
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                VectorizedMinuteToHourEnv(
                    self._two_action_split(names),
                    MinuteToHourEnvConfig(num_envs=1, episode_length=5, cash_idle_penalty_bps=bad, constraints=base),
                    device)

    def test_train_minute_to_hour_validates_constraints_before_transition_table(self) -> None:
        # The trainer validates constraints at ENTRY, before build_transition_feature_table (model inputs)
        # consumes cash_index / count_etf -- a malformed constraint fails before model construction.
        import dataclasses

        from rl_quant.core import DQNLearningConfig
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit, MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        def make_split(name: str) -> HourFromMinuteDataSplit:
            return HourFromMinuteDataSplit(
                name=name, decision_timestamps=["2026-01-02T14:30:00+00:00", "2026-01-03T14:30:00+00:00"],
                next_timestamps=["2026-01-02T15:30:00+00:00", "2026-01-03T15:30:00+00:00"],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, 2)),
                action_valid_mask=torch.ones((2, 2), dtype=torch.bool), label_valid_mask=torch.ones((2, 2), dtype=torch.bool),
                valid_start_indices=torch.tensor([0]), valid_index_mask=torch.tensor([True, False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        learning = DQNLearningConfig(
            num_envs=1, episode_length=2, replay_capacity=8, batch_size=2, train_steps=1, warmup_steps=0,
            gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=1, epsilon_start=0.0,
            epsilon_end=0.0, eval_interval=1, grad_clip=1.0,
        )
        bad_env = MinuteToHourEnvConfig(
            num_envs=1, episode_length=2,
            constraints=dataclasses.replace(default_minute_to_hour_constraints(), count_etf_to_etf_as_two_legs="false"))
        config = MinuteToHourTrainingConfig(
            env=bad_env, learning=learning, use_transition_features=True,
            d_model=8, n_heads=1, minute_layers=1, hour_layers=1, feedforward_dim=8, action_embedding_dim=2,
        )
        with self.assertRaises(ValueError):
            train_minute_to_hour_dqn(make_split("train"), make_split("val"), device=torch.device("cpu"), config=config)

    def test_minute_to_hour_step_fallback_prefers_cash_when_not_first(self) -> None:
        # When a requested action is masked and CASH is NOT the first valid column, the env de-risks to CASH
        # (cash_index), not the first valid action. argmax alone would pick the first valid (e.g. SPY).
        import dataclasses

        from rl_quant.datasets.hour_from_subhour import HourFromMinuteDataSplit, default_minute_to_hour_constraints
        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig, VectorizedMinuteToHourEnv

        device = torch.device("cpu")
        names = ["QQQ", "SPY", "CASH"]  # CASH at index 2 (not first)
        split = HourFromMinuteDataSplit(
            name="t", decision_timestamps=["2026-01-02T10:30:00+00:00", "2026-01-02T11:30:00+00:00"],
            next_timestamps=["2026-01-02T11:30:00+00:00", "2026-01-02T12:30:00+00:00"],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=names,
            minute_features=torch.zeros((2, 1, 1, 1)), minute_mask=torch.ones((2, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((2, 1, 1)), action_returns=torch.zeros((2, 3)),
            action_valid_mask=torch.tensor([[False, True, True], [True, True, True]]),  # QQQ invalid at row 0
            label_valid_mask=torch.ones((2, 3), dtype=torch.bool),
            valid_start_indices=torch.tensor([0]), valid_index_mask=torch.tensor([True, False]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
        )
        cons = dataclasses.replace(default_minute_to_hour_constraints(), cash_index=2, min_hold_bars=0,
                                   cooldown_bars=0, max_switches_per_day=None)
        env = VectorizedMinuteToHourEnv(
            split, MinuteToHourEnvConfig(num_envs=1, episode_length=5, initial_action=2, constraints=cons), device)
        env.reset()
        out = env.step(torch.tensor([0]))  # request QQQ (masked invalid) -> fallback must be CASH(2), not SPY(1)
        self.assertEqual(int(out["actions"][0].item()), 2)

    def test_decision_log_reportability_gate(self) -> None:
        # Additive, LABEL-ONLY reportability validator (moves no P&L). Tiered (base vs strict real-executable)
        # + semantic (finite/sign/equity/ordering, defensive). Aligned to docs/decision_tensor_protocol.md.
        from rl_quant.reportability import (
            REQUIRED_DECISION_LOG_FIELDS,
            evaluate_decision_log_reportability,
        )

        def base_row(**overrides):
            # A complete, semantically-valid BASE row: every protocol field present + valid; numeric
            # timestamps in non-decreasing causal order; positive equity; selected action allowed by the
            # mask. (entry/exit_price are STRICT-tier.)
            row = dict.fromkeys(REQUIRED_DECISION_LOG_FIELDS, 0)
            row.update(
                context_available_until=0, decision_ts=1, entry_execution_ts=2, reward_end_ts=3, exit_execution_ts=4,
                previous_action="CASH", selected_action="QQQ", target_weight=1.0, order_legs=1.0, traded_notional=1.0,
                q_values={"QQQ": 0.1}, q_edge_vs_cash=0.1, q_edge_vs_current=0.1,
                action_mask={"CASH": True, "QQQ": True}, mask_reasons={}, data_quality_score=1.0, readiness_score=1.0,
                gross_return=0.01, cost_bps=1.0, net_return=0.009, equity_after=1.0,
            )
            row.update(overrides)
            return row

        def cats(v):
            return {issue.category for issue in v.issues}

        # Empty logs are not reportable, with a stable token (not ":None").
        v = evaluate_decision_log_reportability([], require_real_executable=False)
        self.assertFalse(v.reportable)
        self.assertIn("missing:decision_rows", v.missing_reportability_reasons)
        # A non-mapping row fails gracefully (no raise).
        self.assertFalse(evaluate_decision_log_reportability(["bad", None, []], require_real_executable=False).reportable)

        # A missing required field fails the base gate.
        v = evaluate_decision_log_reportability([base_row(net_return=None)], require_real_executable=False)
        self.assertFalse(v.reportable)
        self.assertIn("missing:net_return", v.missing_reportability_reasons)

        # Semantic validity (not just presence): NaN cost, negative legs, non-positive equity.
        self.assertIn("malformed", cats(evaluate_decision_log_reportability([base_row(cost_bps=float("nan"))], require_real_executable=False)))
        self.assertIn("negative", cats(evaluate_decision_log_reportability([base_row(order_legs=-1.0)], require_real_executable=False)))
        self.assertIn("nonpositive_equity", cats(evaluate_decision_log_reportability([base_row(equity_after=0.0)], require_real_executable=False)))
        # Malformed turnover is a HARD failure (not silently treated as not-traded).
        self.assertFalse(evaluate_decision_log_reportability([base_row(order_legs=float("nan"))], require_real_executable=False).reportable)
        # The selected action must be allowed by the ex-ante action mask.
        self.assertIn("mask", cats(evaluate_decision_log_reportability([base_row(action_mask={"CASH": True, "QQQ": False})], require_real_executable=False)))

        # Point-in-time-causal timestamp ordering, now PARSED (numeric, ISO-8601, datetime) and enforced --
        # not skipped. context_available_until must precede decision_ts.
        self.assertIn("ordering", cats(evaluate_decision_log_reportability([base_row(exit_execution_ts=0)], require_real_executable=False)))
        self.assertIn("ordering", cats(evaluate_decision_log_reportability([base_row(context_available_until=2, decision_ts=1)], require_real_executable=False)))
        iso = dict(context_available_until="2026-01-02T14:29:59+00:00", decision_ts="2026-01-02T14:30:00+00:00",
                   entry_execution_ts="2026-01-02T14:30:05+00:00", reward_end_ts="2026-01-02T15:30:00+00:00",
                   exit_execution_ts="2026-01-02T15:30:05+00:00")
        self.assertTrue(evaluate_decision_log_reportability([base_row(**iso)], require_real_executable=False).reportable)  # ordered ISO passes
        self.assertIn("ordering", cats(evaluate_decision_log_reportability(  # ISO out of order fails
            [base_row(**{**iso, "exit_execution_ts": "2020-01-01T00:00:00+00:00"})], require_real_executable=False)))

        # Close-only row: base-reportable, but NOT real-executable; strict gaps surfaced regardless.
        close_only = base_row(entry_price=None, exit_price=None)
        v = evaluate_decision_log_reportability([close_only], require_real_executable=False)
        self.assertTrue(v.reportable)
        self.assertTrue(v.base_reportable)
        self.assertFalse(v.real_executable_trade_reportable)
        for tag in ("strict:real_executable_fill_model", "strict:valuation_complete", "strict:execution_complete",
                    "strict:impact_applied", "strict:entry_price"):
            self.assertIn(tag, v.missing_reportability_reasons)
        # Requiring real-executability on the close-only row fails the overall gate.
        self.assertFalse(evaluate_decision_log_reportability([close_only], require_real_executable=True).reportable)

        # Strict fill prices must be FINITE POSITIVE, not merely present (None/NaN/0/negative/str all fail).
        for bad_price in (None, float("nan"), float("inf"), 0.0, -1.0, "100.1"):
            sv = evaluate_decision_log_reportability(
                [base_row(real_executable_fill_model=True, valuation_complete=True, execution_complete=True,
                          impact_applied=True, entry_price=bad_price)],
                require_real_executable=True,
            )
            self.assertFalse(sv.real_executable_trade_reportable, bad_price)
            self.assertIn("strict:entry_price", sv.missing_reportability_reasons)

        # A fully real-executable row passes the strict claim even when required, with no issues.
        real_row = base_row(
            real_executable_fill_model=True, valuation_complete=True, execution_complete=True,
            impact_applied=True, entry_price=100.1, requires_exit_price=True, exit_price=99.9,
        )
        v = evaluate_decision_log_reportability([real_row], require_real_executable=True)
        self.assertTrue(v.reportable)
        self.assertTrue(v.real_executable_trade_reportable)
        self.assertEqual(v.issues, ())
        # Drop the exit price on a row that requires it -> strict claim fails.
        v = evaluate_decision_log_reportability([{**real_row, "exit_price": None}], require_real_executable=True)
        self.assertFalse(v.real_executable_trade_reportable)
        self.assertIn("strict:exit_price", v.missing_reportability_reasons)

    def test_statistical_credibility_psr_dsr(self) -> None:
        # Statistical-credibility axis (separate from mechanical reportability): PSR / expected-max-Sharpe /
        # DSR. Pure formulas (Bailey & Lopez de Prado); change no backtest number. Reference values are
        # hand-computed to pin the formula against transcription errors.
        from rl_quant.statistical_credibility import (
            deflated_sharpe_ratio,
            expected_maximum_sharpe,
            probabilistic_sharpe_ratio,
        )

        # PSR(observed==benchmark) == 0.5 (z=0); monotone in observed_sharpe; -> 1 as observed >> benchmark.
        self.assertAlmostEqual(probabilistic_sharpe_ratio(0.3, benchmark_sharpe=0.3, n_observations=50), 0.5, places=9)
        self.assertLess(
            probabilistic_sharpe_ratio(0.05, benchmark_sharpe=0.0, n_observations=100),
            probabilistic_sharpe_ratio(0.15, benchmark_sharpe=0.0, n_observations=100),
        )
        # Hand-computed: z = 0.1*sqrt(99)/sqrt(1.005) ~= 0.99251 -> Phi ~= 0.8395.
        self.assertAlmostEqual(
            probabilistic_sharpe_ratio(0.1, benchmark_sharpe=0.0, n_observations=100), 0.8395, places=3
        )
        # More observations of the same edge -> more confident.
        self.assertLess(
            probabilistic_sharpe_ratio(0.1, benchmark_sharpe=0.0, n_observations=50),
            probabilistic_sharpe_ratio(0.1, benchmark_sharpe=0.0, n_observations=500),
        )

        # Expected max Sharpe under the null: 0 for one trial, strictly increasing in the number of trials,
        # and scaling linearly with the cross-trial Sharpe std.
        self.assertEqual(expected_maximum_sharpe(1), 0.0)
        self.assertLess(expected_maximum_sharpe(10), expected_maximum_sharpe(1000))
        self.assertGreater(expected_maximum_sharpe(100), 0.0)
        self.assertAlmostEqual(
            expected_maximum_sharpe(100, trials_sharpe_std=2.0), 2.0 * expected_maximum_sharpe(100), places=9
        )

        # DSR with a single trial reduces to PSR vs 0; deflation LOWERS it as the trial count rises.
        self.assertAlmostEqual(
            deflated_sharpe_ratio(0.2, n_trials=1, n_observations=200),
            probabilistic_sharpe_ratio(0.2, benchmark_sharpe=0.0, n_observations=200),
            places=9,
        )
        self.assertGreater(
            deflated_sharpe_ratio(0.2, n_trials=2, n_observations=200, trials_sharpe_std=0.1),
            deflated_sharpe_ratio(0.2, n_trials=500, n_observations=200, trials_sharpe_std=0.1),
        )
        # Validation: bad n_observations / n_trials fail closed.
        for bad in ({"observed_sharpe": 0.1, "benchmark_sharpe": 0.0, "n_observations": 1},):
            with self.assertRaises(ValueError):
                probabilistic_sharpe_ratio(**bad)
        with self.assertRaises(ValueError):
            expected_maximum_sharpe(0)

    def test_protocol_model_input_label_split_validator(self) -> None:
        # Protocol layer (architecture Phase 2): the reusable anti-leakage validator. Enforces the contract a
        # label/future field must NEVER be a model input. Pure/additive (changes no data or training).
        from rl_quant.protocol import (
            assert_no_model_input_leakage,
            validate_decision_tensor_payload,
            validate_model_input_label_split,
        )

        # The canonical second-context split (mirrors features/stock_second_context.py:832-861) must validate.
        model_input_keys = [
            "market_context", "market_context_mask", "action_features", "decision_action_valid_mask",
            "action_valid_mask", "action_cost_bps", "action_target_weights", "portfolio_state",
            "constraint_state", "decision_quality_score", "force_cash_mask",
        ]
        label_keys = [
            "action_returns", "label_valid_mask", "entry_fill_observed_mask", "reward_exit_observed_mask",
            "next_timestamps", "entry_execution_timestamps_ms", "exit_execution_timestamps_ms",
        ]
        forbidden = [
            "action_returns", "label_valid_mask", "entry_fill_observed_mask", "reward_exit_observed_mask",
            "next_timestamps", "exit_execution_timestamps_ms",
        ]
        ok, issues = validate_model_input_label_split(
            model_input_keys=model_input_keys, label_keys=label_keys, forbidden_model_input_keys=forbidden
        )
        self.assertTrue(ok, issues)
        self.assertEqual(issues, ())

        # A leaked label/future field as a model input is rejected (the hard anti-leakage rule).
        leaked_ok, leaked_issues = validate_model_input_label_split(
            model_input_keys=[*model_input_keys, "action_returns"], label_keys=label_keys, forbidden_model_input_keys=forbidden
        )
        self.assertFalse(leaked_ok)
        self.assertTrue(any("leak" in i for i in leaked_issues))
        with self.assertRaises(ValueError):
            assert_no_model_input_leakage(
                model_input_keys=[*model_input_keys, "action_returns"], label_keys=label_keys, forbidden_model_input_keys=forbidden
            )
        # assert_* does not raise on the clean canonical split.
        assert_no_model_input_leakage(
            model_input_keys=model_input_keys, label_keys=label_keys, forbidden_model_input_keys=forbidden
        )

        # Other contract violations: a forbidden key not declared as a label, and an empty model-input list.
        self.assertFalse(validate_model_input_label_split(
            model_input_keys=model_input_keys, label_keys=label_keys,
            forbidden_model_input_keys=[*forbidden, "some_future_flag"],
        )[0])
        self.assertFalse(validate_model_input_label_split(
            model_input_keys=[], label_keys=label_keys, forbidden_model_input_keys=forbidden
        )[0])

        # Payload entry pulls the lists from the payload, falling back to the manifest (like the builder guard).
        ok_payload, _ = validate_decision_tensor_payload(
            {"model_input_keys": model_input_keys, "label_keys": label_keys, "forbidden_model_input_keys": forbidden}
        )
        self.assertTrue(ok_payload)
        ok_manifest, _ = validate_decision_tensor_payload(
            {"model_input_keys": model_input_keys},  # missing the rest in payload...
            {"label_keys": label_keys, "forbidden_model_input_keys": forbidden},  # ...supplied by manifest
        )
        self.assertTrue(ok_manifest)

    def test_flag_registry_governance(self) -> None:
        # Governance: every opt-in flag in the registry is well-formed and defaults OFF (default-preserving),
        # every result-moving flag carries A/B metrics + flip/delete criteria, and the recorded defaults match
        # the actual config dataclass (drift guard) so the registry can't silently disagree with the code.
        import dataclasses

        from rl_quant.envs.minute_to_hour import MinuteToHourEnvConfig
        from rl_quant.protocol.flags import FLAG_REGISTRY, FlagSpec, result_moving_flags
        from rl_quant.training.minute_to_hour import MinuteToHourTrainingConfig

        for name, spec in FLAG_REGISTRY.items():
            self.assertIsInstance(spec, FlagSpec)
            self.assertEqual(spec.name, name)
            self.assertFalse(spec.default, f"{name}: opt-in flags must default to False (default-preserving)")
            if spec.is_result_moving:
                self.assertTrue(spec.required_ab, f"{name}: a result-moving flag must declare A/B metrics")
                self.assertTrue(spec.flip_criterion, f"{name}: missing flip_criterion")
                self.assertTrue(spec.delete_criterion, f"{name}: missing delete_criterion")

        # Result-moving flags: the dynamic + transition model-input flags and the (declared-ahead) training
        # flip use_execution_env_reward. The shadow flag execution_env_reward_shadow is label-changing only
        # (moves metrics/manifest, not P&L), so it is NOT result-moving.
        self.assertEqual(
            set(result_moving_flags()),
            {"use_dynamic_transition_features", "use_transition_features", "use_execution_env_reward"},
        )
        self.assertFalse(FLAG_REGISTRY["execution_env_reward_shadow"].is_result_moving)
        config_defaults = {f.name: f.default for f in dataclasses.fields(MinuteToHourTrainingConfig)}
        for name in ("use_dynamic_transition_features", "use_transition_features"):
            self.assertEqual(
                config_defaults[name], FLAG_REGISTRY[name].default,
                f"{name}: registry default disagrees with MinuteToHourTrainingConfig",
            )
        # execution_env_reward_shadow is a real MinuteToHourEnvConfig field now (PR-3); its registry default
        # must match the config default.
        env_defaults = {f.name: f.default for f in dataclasses.fields(MinuteToHourEnvConfig)}
        self.assertEqual(
            env_defaults["execution_env_reward_shadow"], FLAG_REGISTRY["execution_env_reward_shadow"].default
        )
        # Drift guard: every boolean gating field on the env/training configs must be registered (or explicitly
        # allowlisted as non-governance), so a new result/label-moving flag cannot be wired without a registry
        # entry + its A/B metadata.
        non_governance_bools: set[str] = set()  # add genuinely-cosmetic bools here if ever needed
        config_bools = {
            f.name
            for cfg in (MinuteToHourEnvConfig, MinuteToHourTrainingConfig)
            for f in dataclasses.fields(cfg)
            if f.type in (bool, "bool")
        }
        unregistered = config_bools - set(FLAG_REGISTRY) - non_governance_bools
        self.assertEqual(unregistered, set(), f"unregistered governance flag(s): {unregistered}")

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

    def test_dynamic_transition_features_train_end_to_end_and_artifacts(self) -> None:
        # PR-D D2/D3b: with use_dynamic_transition_features=True the env->replay->forward(dynamic_state)
        # wiring trains end-to-end (shapes line up through rollout, current-Q, and the TD next-state forwards)
        # and the artifact records the dynamic schema. Default off keeps the artifact byte-identical (no
        # dynamic keys / legacy model_version) -- the existing trainer tests cover the off path numerically.
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
            train_minute_to_hour_dqn,
        )
        from rl_quant.trading_constraints import (
            DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION,
            DYNAMIC_TRANSITION_FEATURE_DIM,
            DYNAMIC_TRANSITION_FEATURE_NAMES,
            DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION,
        )

        def make_split(name: str, dates: list[str]) -> HourFromMinuteDataSplit:
            n = len(dates)
            returns = torch.zeros((n, 2))
            returns[:, 1] = 0.01  # non-trivial QQQ return so the dynamic P&L-excursion state is non-degenerate
            return HourFromMinuteDataSplit(
                name=name,
                decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
                next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((n, 1, 1)), action_returns=returns,
                action_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                valid_start_indices=torch.arange(n - 1, dtype=torch.long),
                valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1),
                hours_lookback=1, minutes_per_hour=1,
            )

        train = make_split("train", ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-02", "2026-05-20"])
        val = make_split("val", ["2026-06-01", "2026-06-02"])
        learning = DQNLearningConfig(
            num_envs=2, episode_length=3, replay_capacity=64, batch_size=4, train_steps=8, warmup_steps=2,
            gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=3, epsilon_start=0.2,
            epsilon_end=0.0, eval_interval=4, grad_clip=1.0, use_amp=False,
        )

        def run(dynamic: bool) -> dict:
            config = MinuteToHourTrainingConfig(
                env=MinuteToHourEnvConfig(num_envs=2, episode_length=3), learning=learning,
                d_model=16, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16, action_embedding_dim=4,
                use_dynamic_transition_features=dynamic,
            )
            torch.manual_seed(0)
            _, artifacts = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config)
            return artifacts

        # Flag ON: trains end-to-end (no shape error through the dynamic-threaded forwards) + stamps schema.
        on = run(True)
        self.assertTrue(on["uses_dynamic_transition_features"])
        self.assertEqual(on["model_version"], DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION)
        self.assertEqual(on["dynamic_transition_feature_names"], list(DYNAMIC_TRANSITION_FEATURE_NAMES))
        self.assertEqual(on["dynamic_transition_feature_dim"], DYNAMIC_TRANSITION_FEATURE_DIM)
        self.assertEqual(on["dynamic_transition_feature_schema_version"], DYNAMIC_TRANSITION_FEATURE_SCHEMA_VERSION)
        # Flag OFF (default): the dynamic schema is absent and the model_version is the legacy contract.
        off = run(False)
        self.assertFalse(off["uses_dynamic_transition_features"])
        self.assertEqual(off["dynamic_transition_feature_names"], [])
        self.assertEqual(off["dynamic_transition_feature_dim"], 0)
        self.assertNotEqual(off["model_version"], DYNAMIC_POSITION_AWARE_POLICY_MODEL_VERSION)

        # Clean A/B perturbation: building the zero-init dynamic submodule restores the construction RNG, so
        # flag-on shares flag-off's backbone init -> the FIRST optimizer step is identical (the dynamic head
        # contributes 0 until trained), then the traces DIVERGE once the dynamic encoder receives gradient
        # (the feature actually engages). This is the property a D4 A/B relies on to isolate the feature.
        self.assertTrue(len(on["loss_trace"]) > 1 and len(off["loss_trace"]) == len(on["loss_trace"]))
        self.assertAlmostEqual(on["loss_trace"][0], off["loss_trace"][0], places=6)
        self.assertNotEqual(on["loss_trace"], off["loss_trace"])

    def test_minute_to_hour_baseline_panel(self) -> None:
        # Baseline panel (eval-only; changes no training/reward): deterministic cash / buy-and-hold references
        # run through the SAME eval path as a trained model, so a policy (or a PR-D A/B) must beat them under
        # cost. Single-slot action space -> no equal-weight; always_cash + per-action buy-and-hold.
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            MinuteToHourEvaluationResult,
            evaluate_minute_to_hour_baselines,
        )

        n = 6
        returns = torch.zeros((n, 2))
        returns[:, 1] = 0.01  # QQQ earns +1%/bar; CASH earns 0
        data = HourFromMinuteDataSplit(
            name="val",
            decision_timestamps=[f"2026-06-1{2}T1{4 + i}:30:00+00:00" for i in range(n)],
            next_timestamps=[f"2026-06-1{2}T1{5 + i}:30:00+00:00" for i in range(n)],
            minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1)), action_returns=returns,
            action_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
            valid_start_indices=torch.arange(n - 1, dtype=torch.long),
            valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
            minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1),
            hours_lookback=1, minutes_per_hour=1,
        )
        panel = evaluate_minute_to_hour_baselines(data, device=torch.device("cpu"))

        self.assertIn("always_cash", panel)
        self.assertIn("buy_and_hold:QQQ", panel)
        for result in panel.values():
            self.assertIsInstance(result, MinuteToHourEvaluationResult)
            self.assertTrue(math.isfinite(result.total_return))
        # Cash does nothing (no trades, no cost) -> ~0 return and 0 switches.
        self.assertAlmostEqual(panel["always_cash"].total_return, 0.0, places=6)
        self.assertEqual(panel["always_cash"].allocation_switches, 0)
        # Buy-and-hold QQQ enters once (one switch) and rides the +1%/bar series -> strictly beats cash.
        self.assertEqual(panel["buy_and_hold:QQQ"].allocation_switches, 1)
        self.assertGreater(panel["buy_and_hold:QQQ"].total_return, panel["always_cash"].total_return)

    def test_minute_to_hour_training_state_resumes_from_checkpoint(self) -> None:
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
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

        train = make_split("train", ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
        val = make_split("val", ["2026-02-01", "2026-02-02"])
        env = MinuteToHourEnvConfig(num_envs=2, episode_length=2)

        def config(train_steps: int, state_path: Path, *, resume: bool) -> MinuteToHourTrainingConfig:
            return MinuteToHourTrainingConfig(
                env=env,
                learning=DQNLearningConfig(
                    num_envs=2,
                    episode_length=2,
                    replay_capacity=16,
                    batch_size=2,
                    train_steps=train_steps,
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
                ),
                d_model=16,
                n_heads=2,
                minute_layers=1,
                hour_layers=1,
                feedforward_dim=16,
                action_embedding_dim=4,
                resume_training_state=state_path if resume else None,
                checkpoint_training_state=state_path,
                checkpoint_every_steps=1,
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "training_state.pt"
            torch.manual_seed(123)
            _, first = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config(3, state_path, resume=False))
            self.assertFalse(first["resume"]["loaded"])
            self.assertTrue(state_path.exists())
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["step"], 3)

            _, resumed = train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config(5, state_path, resume=True))
            self.assertTrue(resumed["resume"]["loaded"])
            self.assertEqual(resumed["resume"]["resumed_from_step"], 3)
            self.assertEqual(resumed["resume"]["start_step"], 4)
            saved_again = torch.load(state_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved_again["step"], 5)

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

    def test_episode_terminal_out_of_range_next_action_mask_is_safe_dummy(self) -> None:
        module = __import__(
            "rl_quant.minute_to_hour_transformer",
            fromlist=["VectorizedMinuteToHourEnv", "MinuteToHourEnvConfig", "HourFromMinuteDataSplit"],
        )
        n = 3
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
            valid_start_indices=torch.arange(n, dtype=torch.long),
            valid_index_mask=torch.ones(n, dtype=torch.bool),
            minute_feature_mean=torch.zeros(1),
            minute_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1),
            hour_feature_std=torch.ones(1),
            hours_lookback=1,
            minutes_per_hour=1,
        )
        env = module.VectorizedMinuteToHourEnv(
            split, module.MinuteToHourEnvConfig(num_envs=1, episode_length=10, initial_action=0), torch.device("cpu")
        )
        cash = torch.zeros(1, dtype=torch.long)
        env.indices[:] = n - 1
        transition = env.step(cash)
        self.assertEqual(int(transition["next_indices"][0].item()), n)
        self.assertEqual(float(transition["terminated"][0].item()), 1.0)
        self.assertEqual(float(transition["resets"][0].item()), 1.0)
        self.assertEqual(transition["next_action_mask"].tolist(), [[True, False]])

    def test_calendar_holdout_trainer_filters_partitions_before_boundaries(self) -> None:
        script_path = ROOT / "scripts" / "train_hourly_from_second_calendar_holdout.py"
        spec = importlib.util.spec_from_file_location("calendar_holdout_trainer", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in [
                "2026-01-01_to_2026-01-04",
                "2026-02-01_to_2026-02-04",
                "2026-06-10_to_2026-06-13",
            ]:
                partition_dir = root / label
                partition_dir.mkdir(parents=True)
                (partition_dir / "hour_from_second_dataset.pt").touch()
            args = module.parse_args(
                [
                    "--partitions-root",
                    str(root),
                    "--max-partitions",
                    "2",
                    "--partition-selection",
                    "latest",
                    "--test-months",
                    "2",
                    "--val-months",
                    "1",
                ]
            )
            paths = module.partition_paths(args)
            self.assertEqual(
                [path.parent.name for path in paths],
                ["2026-02-01_to_2026-02-04", "2026-06-10_to_2026-06-13"],
            )
            boundaries = module.calendar_boundaries(args, paths)
            self.assertEqual(boundaries["train_end_ts"], "2026-03-13T00:00:00+00:00")
            self.assertEqual(boundaries["val_end_ts"], "2026-04-13T00:00:00+00:00")
            self.assertEqual(boundaries["test_start_ts"], "2026-04-13T00:00:00+00:00")
            self.assertEqual(boundaries["test_end_ts"], "2026-06-13T00:00:00+00:00")

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
        # Device equality is validated so a CPU/GPU mix fails legibly here, not deep inside torch.where.
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                dqn_td_target(rewards, 0.9, terminated, next_q.cuda())
        # A non-finite CHOSEN reward is rejected -- unlike a terminal next_q it is never discarded, so a
        # NaN/Inf reward (an unmasked invalid action return) would silently poison the target. Rejected
        # on BOTH non-terminal and terminal rows, since the reward term is always added.
        with self.assertRaises(ValueError):
            dqn_td_target(torch.tensor([1.0, float("nan")]), 0.9, torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        with self.assertRaises(ValueError):
            dqn_td_target(torch.tensor([float("inf"), 1.0]), 0.9, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

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

    def test_dqn_td_target_rejects_invalid_gamma(self) -> None:
        from rl_quant.core import dqn_td_target

        rewards, terminated, next_q = torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([5.0])
        for bad_gamma in (1.5, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                dqn_td_target(rewards, bad_gamma, terminated, next_q)
        self.assertAlmostEqual(float(dqn_td_target(rewards, 0.9, terminated, next_q)[0].item()), 1.0 + 0.9 * 5.0, places=5)

    def test_dqn_td_target_rejects_non_binary_terminated_and_nonfinite_next_q(self) -> None:
        from rl_quant.core import dqn_td_target

        # Non-binary terminated mask is a corruption, not a 50% terminal.
        with self.assertRaises(ValueError):
            dqn_td_target(torch.tensor([1.0]), 0.99, torch.tensor([0.5]), torch.tensor([2.0]))
        # A NON-terminal row with non-finite next_q is rejected; a TERMINAL row's NaN next_q is fine.
        with self.assertRaises(ValueError):
            dqn_td_target(torch.tensor([1.0]), 0.99, torch.tensor([0.0]), torch.tensor([float("nan")]))
        target = dqn_td_target(torch.tensor([1.0]), 0.99, torch.tensor([1.0]), torch.tensor([float("nan")]))
        self.assertAlmostEqual(float(target[0].item()), 1.0, places=5)

    def test_safe_next_row_indices_clamps_terminal_but_rejects_nonterminal_oob(self) -> None:
        from rl_quant.core import as_binary_bool_mask, safe_next_row_indices

        # Terminal out-of-range indices clamp into [min_index, max_index]; crucially they clamp to
        # MIN_INDEX (here 2), not 0, so a clamped terminal dummy never builds a tail-wrapped window.
        out = safe_next_row_indices(
            torch.tensor([7, 0, 3]), torch.tensor([1.0, 1.0, 0.0]), min_index=2, max_index=5
        )
        self.assertEqual(out.tolist(), [5, 2, 3])  # 7->5 (max); 0->2 (min, terminal); 3 in range
        # A NON-terminal index below min_index (would wrap the lookback window) is rejected.
        with self.assertRaises(ValueError):
            safe_next_row_indices(torch.tensor([1, 3]), torch.tensor([0.0, 0.0]), min_index=2, max_index=5)
        # A NON-terminal index above max_index is rejected too.
        with self.assertRaises(ValueError):
            safe_next_row_indices(torch.tensor([6, 3]), torch.tensor([0.0, 0.0]), min_index=2, max_index=5)
        # valid_index_mask rejects an in-range-but-INVALID non-terminal row (row 4 is invalid here)...
        mask = torch.tensor([True, True, True, True, False, True])
        with self.assertRaises(ValueError):
            safe_next_row_indices(
                torch.tensor([4, 3]), torch.tensor([0.0, 0.0]), min_index=2, max_index=5, valid_index_mask=mask
            )
        # ...but in-range VALID non-terminal rows pass through unchanged.
        ok = safe_next_row_indices(
            torch.tensor([3, 5]), torch.tensor([0.0, 0.0]), min_index=2, max_index=5, valid_index_mask=mask
        )
        self.assertEqual(ok.tolist(), [3, 5])
        # A TERMINAL row may sit on an invalid/out-of-range index without tripping the mask check.
        term_ok = safe_next_row_indices(
            torch.tensor([4, 99]), torch.tensor([1.0, 1.0]), min_index=2, max_index=5, valid_index_mask=mask
        )
        self.assertEqual(term_ok.tolist(), [4, 5])
        # Bad min/max ordering, a non-bool mask, and a too-short mask are all rejected.
        with self.assertRaises(ValueError):
            safe_next_row_indices(torch.tensor([3]), torch.tensor([0.0]), min_index=5, max_index=2)
        with self.assertRaises(ValueError):
            safe_next_row_indices(
                torch.tensor([3]), torch.tensor([0.0]), min_index=0, max_index=5,
                valid_index_mask=torch.ones(6, dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            safe_next_row_indices(
                torch.tensor([3]), torch.tensor([0.0]), min_index=0, max_index=5,
                valid_index_mask=torch.ones(3, dtype=torch.bool),
            )
        # Shape mismatch must be rejected up front (else the &-mask broadcasts to the wrong rows).
        with self.assertRaises(ValueError):
            safe_next_row_indices(torch.tensor([[1], [2]]), torch.tensor([0.0, 0.0]), min_index=0, max_index=4)
        # A float "index" tensor is rejected -- indices must be torch.long.
        with self.assertRaises(ValueError):
            safe_next_row_indices(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), min_index=0, max_index=4)
        # as_binary_bool_mask: bool passthrough; binary float ok; non-binary rejected.
        self.assertTrue(bool(as_binary_bool_mask(torch.tensor([True, False]))[0].item()))
        self.assertEqual(as_binary_bool_mask(torch.tensor([1.0, 0.0])).tolist(), [True, False])
        with self.assertRaises(ValueError):
            as_binary_bool_mask(torch.tensor([0.5]))

    def test_replay_buffers_validate_batch_shapes(self) -> None:
        from rl_quant.core import TensorDictReplayBuffer, TensorReplayBuffer

        cpu = torch.device("cpu")
        buf = TensorReplayBuffer(capacity=8, device=cpu, fields={"a": torch.float32, "b": torch.long})
        buf.add(a=torch.zeros(3), b=torch.zeros(3, dtype=torch.long), extra=torch.zeros(3))  # extra ignored
        with self.assertRaises(ValueError):
            buf.add(a=torch.zeros(3), b=torch.zeros(2, dtype=torch.long))  # mismatched leading batch dim
        dbuf = TensorDictReplayBuffer(capacity=8, device=cpu, fields={"x": ((4,), torch.float32)})
        dbuf.add(x=torch.zeros(3, 4))
        with self.assertRaises(ValueError):
            dbuf.add(x=torch.zeros(3, 5))  # wrong trailing shape

    def test_replay_buffer_extra_first_field_does_not_define_batch_size(self) -> None:
        from rl_quant.core import TensorDictReplayBuffer

        buffer = TensorDictReplayBuffer(
            capacity=8,
            device=torch.device("cpu"),
            fields={"states": ((3,), torch.float32), "actions": ((), torch.long)},
        )
        # An extra field placed FIRST with a different leading dim must NOT define the write size;
        # the canonical count comes from the declared replay fields (2 rows here), not "extra" (99).
        buffer.add(extra=torch.zeros(99), states=torch.zeros(2, 3), actions=torch.zeros(2, dtype=torch.long))
        self.assertEqual(buffer.size, 2)

    def test_autocast_context_validates_dtype_eagerly_and_normalizes(self) -> None:
        from rl_quant.core import autocast_context, resolve_amp_dtype

        # Normalization: whitespace/case-insensitive, accepts long and short spellings.
        self.assertEqual(resolve_amp_dtype("  FP16 "), torch.float16)
        self.assertEqual(resolve_amp_dtype("BFloat16"), torch.bfloat16)
        # Eager validation: a bad dtype is rejected even when AMP is disabled (CPU / requested=False).
        with self.assertRaises(ValueError):
            autocast_context(torch.device("cpu"), False, "fp8")

    def test_cuda_memory_report_round_digits(self) -> None:
        from rl_quant.core import cuda_memory_report

        cpu = torch.device("cpu")
        raw = cuda_memory_report(cpu)
        rounded = cuda_memory_report(cpu, round_digits=2)
        # CPU is all-zero either way; the key contract is that raw (guard) and rounded (log) both work.
        self.assertEqual(set(raw), set(rounded))
        self.assertEqual(raw["free_gb"], 0.0)
        self.assertEqual(rounded["free_gb"], 0.0)

    def test_intraday_training_config_has_amp_dtype(self) -> None:
        import dataclasses

        module = __import__("rl_quant.intraday_dqn", fromlist=["TrainingConfig"])
        fields = {f.name: f for f in dataclasses.fields(module.TrainingConfig)}
        self.assertIn("amp_dtype", fields)
        self.assertEqual(fields["amp_dtype"].default, "fp16")

    def test_intraday_valid_index_mask_matches_valid_start_range(self) -> None:
        import csv as _csv
        import tempfile
        from pathlib import Path as _Path

        from rl_quant.datasets.intraday import _finalize_split, _load_raw_split

        cols = [
            "time", "bucket_start_ns", "bucket_seconds", "close_mid", "best_bid", "best_ask",
            "close_spread", "avg_spread", "close_imbalance", "avg_imbalance", "close_microprice",
            "high_mid", "low_mid", "quote_updates", "bid_depth_lots", "ask_depth_lots",
            "locked_quotes", "crossed_quotes",
        ]
        row = {
            "time": "09:30:00", "bucket_start_ns": "0", "bucket_seconds": "1", "close_mid": "100.0",
            "best_bid": "99.9", "best_ask": "100.1", "close_spread": "0.2", "avg_spread": "0.2",
            "close_imbalance": "0.5", "avg_imbalance": "0.5", "close_microprice": "100.0",
            "high_mid": "100.2", "low_mid": "99.8", "quote_updates": "10", "bid_depth_lots": "5",
            "ask_depth_lots": "5", "locked_quotes": "0", "crossed_quotes": "0",
        }
        with tempfile.TemporaryDirectory() as d:
            path = _Path(d) / "2026-01-02_nbbo_1s.csv"
            with path.open("w", newline="") as fh:
                writer = _csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                for _ in range(6):
                    writer.writerow(row)
            raw = _load_raw_split("train", [path], lookback=2)
        mask = raw["valid_index_mask"]
        # 6 rows, lookback=2 -> valid range [1, 4]; row 0 (< lookback-1) and row 5 (= day_end-1, no
        # in-day finite next) are excluded. The mask must be the FULL valid range, == valid_start_indices.
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(int(mask.shape[0]), 6)
        self.assertEqual(mask.nonzero().flatten().tolist(), [1, 2, 3, 4])
        self.assertEqual(mask.nonzero().flatten().tolist(), raw["valid_start_indices"].tolist())
        self.assertFalse(bool(mask[0].item()))
        self.assertFalse(bool(mask[5].item()))
        # The new required field survives finalize() and a device move (.to keeps it co-located).
        split = _finalize_split(raw, feature_mean=torch.zeros(14), feature_std=torch.ones(14))
        self.assertEqual(split.to(torch.device("cpu")).valid_index_mask.tolist(), mask.tolist())

    def test_all_dqn_trainers_bootstrap_on_terminated_not_resets(self) -> None:
        # The load-bearing RL invariant: the TD bootstrap is masked by `terminated` (a true terminal),
        # NEVER by resets/dones/truncated -- treating an episode-length truncation as terminal would
        # bias values toward short horizons. Lock it across every trainer's dqn_td_target call so a new
        # trainer (or a careless edit) that bootstraps on the reset mask fails the gate.
        import re

        src = ROOT / "src" / "rl_quant"
        trainers = ["training/strategy.py", "training/intraday.py", "training/hourly.py", "training/minute_to_hour.py"]
        call_re = re.compile(r"dqn_td_target\(([^\n]*)\)")
        for name in trainers:
            calls = call_re.findall((src / name).read_text())
            self.assertTrue(calls, f"{name}: expected a dqn_td_target(...) call")
            for args in calls:
                self.assertIn('batch["terminated"]', args, f"{name}: dqn_td_target must bootstrap on terminated")
                for forbidden in ("resets", "dones", "truncated"):
                    self.assertNotIn(forbidden, args, f"{name}: dqn_td_target must not bootstrap on {forbidden}")

    def test_architecture_layer_import_boundaries(self) -> None:
        # Lock the protocol-first layering so it can't erode (Sculley et al., "Hidden Technical Debt in ML
        # Systems": boundary erosion / undeclared consumers). The foundational/low layers must not import
        # higher ones. These rules hold as of the reorg; this guards against regression. TYPE_CHECKING-only
        # imports are excluded (they are not runtime dependencies -- e.g. the envs<->training annotation
        # cycle-break), so a layer importing a higher one purely for typing is allowed.
        import ast

        forbidden_by_layer = {
            "protocol": ("data_sources", "features", "datasets", "envs", "models", "training", "evaluation", "reportability", "workflows"),
            "data_sources": ("features", "datasets", "envs", "models", "training", "evaluation", "reportability", "workflows"),
            "features": ("datasets", "envs", "models", "training", "evaluation", "reportability", "workflows"),
            "datasets": ("envs", "models", "training", "evaluation", "reportability", "workflows"),
            "models": ("datasets", "envs", "training", "evaluation", "reportability", "workflows"),
        }

        def runtime_imports(tree: ast.AST) -> list[str]:
            mods: list[str] = []

            def visit(node: ast.AST) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.If):
                        test = child.test
                        is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                        )
                        if is_type_checking:
                            for n in child.orelse:  # the else branch IS runtime
                                visit(n)
                            continue
                    if isinstance(child, ast.ImportFrom) and child.module:
                        mods.append(child.module)
                    elif isinstance(child, ast.Import):
                        mods.extend(alias.name for alias in child.names)
                    visit(child)

            visit(tree)
            return mods

        src = ROOT / "src" / "rl_quant"
        violations: list[str] = []
        for layer, forbidden in forbidden_by_layer.items():
            for path in sorted((src / layer).rglob("*.py")):
                for module in runtime_imports(ast.parse(path.read_text())):
                    for higher in forbidden:
                        if module == f"rl_quant.{higher}" or module.startswith(f"rl_quant.{higher}."):
                            violations.append(f"{layer}/{path.name} -> {module} (forbidden: {layer} must not import {higher})")
        self.assertEqual(violations, [], "layer import-boundary violations:\n" + "\n".join(violations))

    def test_news_article_rows_reject_negative_source_latency(self) -> None:
        # A negative source latency implies availability BEFORE publish (look-ahead). The library must
        # fail closed at every entry point, not silently clamp to 0 (optimistic) as it once did.
        from pathlib import Path as _Path

        from rl_quant.features.news_llm import (
            _raw_article_row,
            build_news_article_rows,
            write_news_article_outputs,
        )

        with self.assertRaises(ValueError):
            _raw_article_row("AAPL", {"published_utc": "2026-01-02T00:00:00Z"}, 0, source_latency_seconds=-1)
        with self.assertRaises(ValueError):  # validated up front, before any filesystem access
            build_news_article_rows(raw_root=_Path("/does/not/exist"), symbols=["AAPL"], source_latency_seconds=-5)
        with self.assertRaises(ValueError):  # validated before mkdir / parquet write
            write_news_article_outputs(
                rows=[], output_root=_Path("/does/not/exist"), raw_root=_Path("/does/not/exist"),
                symbols=[], errors=[], source_latency_seconds=-1,
            )

    def test_amp_dtype_reaches_minute_to_hour_autocast(self) -> None:
        import unittest.mock as mock

        import rl_quant.training.minute_to_hour as m2h  # the train loop looks up autocast_context here
        from rl_quant.core import DQNLearningConfig
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit,
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
            train_minute_to_hour_dqn,
        )

        def make_split(name: str, dates: list[str]) -> HourFromMinuteDataSplit:
            n = len(dates)
            return HourFromMinuteDataSplit(
                name=name,
                decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
                next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((n, 1, 1)), action_returns=torch.zeros((n, 2)),
                action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                valid_start_indices=torch.arange(n - 1, dtype=torch.long), valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        train = make_split("train", ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-02", "2026-05-20"])
        val = make_split("val", ["2026-06-01", "2026-06-02"])
        learning = DQNLearningConfig(
            num_envs=2, episode_length=3, replay_capacity=64, batch_size=4, train_steps=6, warmup_steps=2,
            gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=3, epsilon_start=0.2,
            epsilon_end=0.0, eval_interval=4, grad_clip=1.0, amp_dtype="bf16",
        )
        config = MinuteToHourTrainingConfig(
            env=MinuteToHourEnvConfig(num_envs=2, episode_length=3), learning=learning,
            d_model=16, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16, action_embedding_dim=4,
        )
        seen: list[str] = []
        real = m2h.autocast_context

        def recorder(device, requested, amp_dtype="fp16"):
            seen.append(amp_dtype)
            return real(device, requested, amp_dtype)

        torch.manual_seed(0)
        with mock.patch.object(m2h, "autocast_context", recorder):
            train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config)
        # The trainer must thread config.learning.amp_dtype into autocast_context (not a hardcoded fp16).
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"bf16"})

    def test_minute_to_hour_execution_shadow_training_byte_identical_and_surfaced(self) -> None:
        # PR-3 end-to-end: execution_env_reward_shadow must NOT change training (the shadow reward is a logged
        # side-channel, never trained on) yet MUST surface the shadow deltas in the artifact. Train a tiny run
        # twice (shadow off vs on, same seed) -> identical loss/reward traces; only the on-run carries the deltas.
        from rl_quant.core import DQNLearningConfig
        from rl_quant.datasets.hour_from_subhour import default_minute_to_hour_constraints
        from rl_quant.minute_to_hour_transformer import (
            HourFromMinuteDataSplit, MinuteToHourEnvConfig, MinuteToHourTrainingConfig, train_minute_to_hour_dqn,
        )

        def make_split(name: str, dates: list[str]) -> HourFromMinuteDataSplit:
            n = len(dates)
            return HourFromMinuteDataSplit(
                name=name, decision_timestamps=[f"{d}T14:30:00+00:00" for d in dates],
                next_timestamps=[f"{d}T15:30:00+00:00" for d in dates],
                minute_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
                minute_features=torch.zeros((n, 1, 1, 1)), minute_mask=torch.ones((n, 1, 1), dtype=torch.bool),
                hour_features=torch.zeros((n, 1, 1)), action_returns=torch.zeros((n, 2)),
                action_valid_mask=torch.ones((n, 2), dtype=torch.bool), label_valid_mask=torch.ones((n, 2), dtype=torch.bool),
                valid_start_indices=torch.arange(n - 1, dtype=torch.long), valid_index_mask=torch.tensor([True] * (n - 1) + [False]),
                minute_feature_mean=torch.zeros(1), minute_feature_std=torch.ones(1),
                hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1), hours_lookback=1, minutes_per_hour=1,
            )

        train = make_split("train", ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-02", "2026-05-20"])
        val = make_split("val", ["2026-06-01", "2026-06-02"])

        def run(shadow: bool) -> dict:
            learning = DQNLearningConfig(
                num_envs=2, episode_length=3, replay_capacity=64, batch_size=4, train_steps=6, warmup_steps=2,
                gamma=0.99, learning_rate=1e-3, weight_decay=0.0, target_update_interval=3, epsilon_start=0.2,
                epsilon_end=0.0, eval_interval=4, grad_clip=1.0,
            )
            config = MinuteToHourTrainingConfig(
                env=MinuteToHourEnvConfig(num_envs=2, episode_length=3, execution_env_reward_shadow=shadow),
                learning=learning, d_model=16, n_heads=2, minute_layers=1, hour_layers=1, feedforward_dim=16,
                action_embedding_dim=4,
            )
            torch.manual_seed(0)
            return train_minute_to_hour_dqn(train, val, device=torch.device("cpu"), config=config)[1]  # artifacts

        off, on = run(False), run(True)
        self.assertEqual(off["loss_trace"], on["loss_trace"])  # training byte-identical (shadow is a side-channel)
        self.assertEqual(off["train_reward_trace"], on["train_reward_trace"])
        self.assertFalse(off["execution_env_reward_shadow"])
        self.assertIsNone(off["execution_shadow_reward_delta_mean"])
        self.assertTrue(on["execution_env_reward_shadow"])
        self.assertIsNotNone(on["execution_shadow_reward_delta_mean"])  # surfaced when on
        # #8: real_executable is None when no shadow ran (distinguishes "no shadow" from "shadow, not real-exec")
        # and explicitly False (never True) when on; the priced FEE is surfaced (and None when off).
        self.assertIsNone(off["execution_shadow_real_executable"])
        self.assertIsNone(off["execution_shadow_fee_bps"])
        self.assertIs(on["execution_shadow_real_executable"], False)
        self.assertEqual(on["execution_shadow_fee_bps"], float(default_minute_to_hour_constraints().one_way_cost_bps))
        # #7 auditability: action-metadata fingerprint + kept-regularizer + weight-semantics fields (None off).
        self.assertIsNone(off["execution_shadow_action_metadata_hash"])
        self.assertIsNone(off["execution_shadow_keeps_switch_penalty"])
        self.assertIsInstance(on["execution_shadow_action_metadata_hash"], str)
        self.assertTrue(on["execution_shadow_action_metadata_complete"])          # CASH + QQQ both known
        self.assertEqual(on["execution_shadow_unknown_action_symbols"], [])
        self.assertIs(on["execution_shadow_keeps_switch_penalty"], True)
        self.assertIs(on["execution_shadow_keeps_cash_idle"], True)
        self.assertEqual(on["execution_shadow_linear_impact_bps_per_weight"], 0.0)
        self.assertIn("UNRESOLVED", on["execution_shadow_weight_semantics_assumed"])

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

    def test_qt_cli_workflows_map_to_existing_scripts(self) -> None:
        from rl_quant.workflows.cli import _DISPATCH
        from rl_quant.paths import scripts_dir

        for (group, workflow), script in _DISPATCH.items():
            path = scripts_dir() / script
            self.assertTrue(path.exists(), f"qt {group} {workflow} -> missing script {path}")

    def test_qt_cli_dispatch_expands_preset_and_forwards_args(self) -> None:
        from rl_quant.cli import build_parser, resolve_workflow
        from rl_quant.paths import scripts_dir

        parser = build_parser()
        # --source 1s selects the second-context preset by default; --foo bar is forwarded verbatim.
        args, passthrough = parser.parse_known_args(["train", "subhour", "--source", "1s", "--foo", "bar"])
        script, script_argv = resolve_workflow(args, passthrough)
        self.assertEqual(script, "train_hourly_from_minute_context_rl.py")
        self.assertTrue((scripts_dir() / script).exists())
        self.assertIn("--run-name", script_argv)  # from the preset
        # Passthrough args come AFTER preset args so the user overrides defaults.
        self.assertEqual(script_argv[-2:], ["--foo", "bar"])
        # No selector default -> no preset, args forwarded unchanged.
        args2, passthrough2 = parser.parse_known_args(["train", "subhour", "--source", "1m", "--x", "1"])
        _, argv2 = resolve_workflow(args2, passthrough2)
        self.assertEqual(argv2, ["--x", "1"])

    def test_qt_preset_commands_and_registry(self) -> None:
        from rl_quant.workflows.cli import _DISPATCH, main
        from rl_quant.presets import PRESETS, resolve_preset

        self.assertEqual(main(["preset", "list"]), 0)
        self.assertEqual(main(["preset", "show", "train.subhour.second-context"]), 0)
        with self.assertRaises(SystemExit):
            resolve_preset("does-not-exist")
        # Every preset expands to a non-empty arg list and targets a real (group, workflow).
        for name, preset in PRESETS.items():
            self.assertTrue(resolve_preset(name), f"preset {name} expanded empty")
            group, workflow = preset.workflow.split(".", 1)
            self.assertIn((group, workflow), _DISPATCH, f"preset {name} targets unknown workflow")

    def test_qt_rejects_preset_for_wrong_workflow(self) -> None:
        from rl_quant.cli import build_parser, resolve_workflow

        parser = build_parser()
        # A direct-bar preset must NOT be accepted for a second-context command (would forward the
        # wrong CLI flags to the script).
        args, passthrough = parser.parse_known_args(
            ["train", "second-context", "--preset", "train.direct-bar.minute"]
        )
        with self.assertRaises(SystemExit):
            resolve_workflow(args, passthrough)
        # The matching preset is accepted.
        args2, pt2 = parser.parse_known_args(["train", "direct-bar", "--preset", "train.direct-bar.minute"])
        script, argv = resolve_workflow(args2, pt2)
        self.assertEqual(script, "train_hourly_causal_transformer_rl.py")
        self.assertTrue(argv)

    def test_minute_to_hour_uses_core_replay_buffer(self) -> None:
        # The local duplicate was removed; the trainer must use core's validated buffer.
        import rl_quant.core as core
        import rl_quant.minute_to_hour_transformer as m2h

        self.assertIs(m2h.TensorDictReplayBuffer, core.TensorDictReplayBuffer)

    def test_runtime_config_add_args_and_resolve(self) -> None:
        import argparse

        from rl_quant.config import RuntimeConfig, add_runtime_args, resolve_runtime

        parser = argparse.ArgumentParser()
        add_runtime_args(parser, seed_default=7)
        # --device cpu so the test is deterministic regardless of CUDA availability.
        args = parser.parse_args(["--device", "cpu", "--amp", "--amp-dtype", "bf16", "--min-free-vram-gb", "4", "--seed", "5"])
        self.assertEqual((args.device, args.amp, args.amp_dtype, args.min_free_vram_gb, args.seed), ("cpu", True, "bf16", 4.0, 5))
        runtime = resolve_runtime(args)
        self.assertIsInstance(runtime, RuntimeConfig)
        self.assertEqual(runtime.device.type, "cpu")
        self.assertTrue(runtime.use_amp)
        self.assertEqual(runtime.amp_dtype, "bf16")
        self.assertEqual(runtime.seed, 5)


if __name__ == "__main__":
    unittest.main()
