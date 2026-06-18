from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from _support import load_script


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
