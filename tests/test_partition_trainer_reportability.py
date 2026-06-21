"""Reportability invariants for the hour-from-second rolling partition trainer.

Pins the Rule-B fix: --skip-failed-partitions keeps a long full-history run alive, but a skipped/failed selected
partition leaves a HOLE in the warm-start lineage, so the official result must NOT be reported as a clean
full-history strict result. These exercise the pure ``aggregate_reportability`` helper (no training run needed).
"""

from __future__ import annotations

import unittest

from tests._support import load_script


class PartitionTrainerReportabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_script("train_hourly_from_second_protocol_partitions")

    @staticmethod
    def _rec(partition: str, *, status: str = "ok", official: bool = False, reportable: bool = True) -> dict:
        return {
            "partition": partition,
            "status": status,
            "is_official_latest_test": official,
            "evaluation_reportable": reportable,
        }

    def test_clean_run_is_reportable(self) -> None:
        # No skips, reportable official partition, latest coverage -> reportable exactly as before (default-preserving).
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-01-01_to_2026-01-04"), self._rec("2026-06-10_to_2026-06-13", official=True)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertTrue(agg["aggregate_reportable"])
        self.assertTrue(agg["official_latest_reportable"])
        self.assertTrue(agg["all_prior_partitions_completed"])
        self.assertEqual(agg["aggregate_reportability_errors"], [])

    def test_skipped_prior_partition_makes_official_non_reportable(self) -> None:
        # The central review P0: a skipped 2022 training partition + an "ok" 2026 official partition must NOT
        # yield a reportable official result -- the lineage has a hole.
        agg = self.mod.aggregate_reportability(
            [
                self._rec("2022-01-20_to_2022-01-25", status="failed"),
                self._rec("2026-06-10_to_2026-06-13", official=True),
            ],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertFalse(agg["official_latest_reportable"])
        self.assertFalse(agg["all_prior_partitions_completed"])
        self.assertIn("2022-01-20_to_2022-01-25", agg["skipped_partition_labels_before_official_test"])
        self.assertTrue(
            any("selected_partitions_failed_or_skipped" in e for e in agg["aggregate_reportability_errors"])
        )

    def test_failed_official_partition_is_non_reportable(self) -> None:
        # If the official latest partition itself failed (skipped), there is no valid official KPI.
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-06-10_to_2026-06-13", status="failed")],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertIn("official_latest_test_did_not_complete", agg["aggregate_reportability_errors"])

    def test_non_latest_official_is_non_reportable(self) -> None:
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-03-01_to_2026-03-04", official=True)],
            official_test_label="2026-03-01_to_2026-03-04",
            final_test_is_latest_available=False,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertIn("final_test_not_latest_available", agg["aggregate_reportability_errors"])

    def test_non_reportable_official_partition_blocks_aggregate(self) -> None:
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-06-10_to_2026-06-13", official=True, reportable=False)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertIn("official_latest_test_non_reportable", agg["aggregate_reportability_errors"])

    def test_skip_flag_defaults_off(self) -> None:
        # Default remains fail-loud: the flag is opt-in.
        self.assertFalse(self.mod.parse_args(["--run-name", "x"]).skip_failed_partitions)
        self.assertTrue(self.mod.parse_args(["--run-name", "x", "--skip-failed-partitions"]).skip_failed_partitions)


if __name__ == "__main__":
    unittest.main()
