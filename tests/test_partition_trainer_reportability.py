"""Reportability + failure-classification invariants for the hour-from-second rolling partition trainer.

Pins the hardened Rule-B behaviour: --skip-failed-partitions keeps a long full-history run alive, but a
skipped/failed/non-reportable selected partition leaves a HOLE (or a tainted window) in the warm-start lineage,
so the official result must NOT be reported as a clean full-history strict result. Also pins the
failed-vs-skipped distinction and the fatal/skippable failure classifier. Exercises pure helpers (no training).
"""

from __future__ import annotations

import unittest

from tests._support import load_script


class PartitionTrainerReportabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_script("train_hourly_from_second_protocol_partitions")

    @staticmethod
    def _rec(
        partition: str,
        *,
        status: str = "ok",
        official: bool = False,
        reportable: bool = True,
        split_reportable: bool = True,
        skipped: bool = False,
    ) -> dict:
        return {
            "partition": partition,
            "status": status,
            "is_official_latest_test": official,
            "evaluation_reportable": reportable,
            "split_reportable": split_reportable,
            "skipped": skipped,
        }

    # ---- aggregate_reportability ----

    def test_clean_run_is_reportable(self) -> None:
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-01-01_to_2026-01-04"), self._rec("2026-06-10_to_2026-06-13", official=True)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertTrue(agg["aggregate_reportable"])
        self.assertTrue(agg["official_latest_reportable"])
        self.assertTrue(agg["official_partition_evaluation_reportable"])
        self.assertTrue(agg["all_prior_partitions_completed"])
        self.assertEqual(agg["aggregate_reportability_errors"], [])

    def test_failed_prior_partition_makes_official_non_reportable(self) -> None:
        # Central P0: a failed 2022 partition + an "ok" 2026 official partition -> official NOT reportable.
        agg = self.mod.aggregate_reportability(
            [self._rec("2022-01-20_to_2022-01-25", status="failed"),
             self._rec("2026-06-10_to_2026-06-13", official=True)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertFalse(agg["official_latest_reportable"])
        self.assertTrue(agg["official_partition_evaluation_reportable"])  # the official partition itself passed
        self.assertFalse(agg["all_prior_partitions_completed"])
        self.assertIn("2022-01-20_to_2022-01-25", agg["failed_partition_labels_before_official_test"])

    def test_completed_but_non_reportable_prior_window_taints_lineage(self) -> None:
        # P0.2: a prior partition that is status=ok but NON-reportable still advanced the warm-start lineage ->
        # the official model trained through a non-reportable window -> not a clean result.
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-01-01_to_2026-01-04", reportable=False),
             self._rec("2026-06-10_to_2026-06-13", official=True)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertIn("2026-01-01_to_2026-01-04", agg["non_reportable_completed_partition_labels"])
        self.assertTrue(any("completed_but_non_reportable" in e for e in agg["aggregate_reportability_errors"]))

    def test_failed_vs_skipped_naming(self) -> None:
        # P1.1: failed == status!=ok (gates reportability); skipped == the subset actually skipped under the flag.
        agg = self.mod.aggregate_reportability(
            [self._rec("2022-a", status="failed", skipped=True),   # data ValueError, skipped under the flag
             self._rec("2022-b", status="failed", skipped=False),  # fatal failure, NOT skipped
             self._rec("2026-06-10_to_2026-06-13", official=True)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertEqual(agg["failed_partition_count"], 2)
        self.assertEqual(agg["skipped_partition_count"], 1)
        self.assertEqual(agg["skipped_partition_labels"], ["2022-a"])

    def test_failed_official_partition_is_non_reportable(self) -> None:
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-06-10_to_2026-06-13", status="failed", official=True)],
            official_test_label="2026-06-10_to_2026-06-13",
            final_test_is_latest_available=True,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertIn("official_latest_test_did_not_complete", agg["aggregate_reportability_errors"])

    def test_non_latest_official_is_non_reportable(self) -> None:
        # P0.3: official partition passes its own gate, but it is not the latest -> official KPI not reportable.
        agg = self.mod.aggregate_reportability(
            [self._rec("2026-03-01_to_2026-03-04", official=True)],
            official_test_label="2026-03-01_to_2026-03-04",
            final_test_is_latest_available=False,
        )
        self.assertFalse(agg["aggregate_reportable"])
        self.assertFalse(agg["official_latest_reportable"])  # bound to the whole-run verdict
        self.assertTrue(agg["official_partition_evaluation_reportable"])  # the partition itself was fine
        self.assertIn("final_test_not_latest_available", agg["aggregate_reportability_errors"])

    def test_ordinal_order_not_label_string_for_before_official(self) -> None:
        # P1.2: "before official" uses record ORDER, not label string comparison (robust to label schemes).
        agg = self.mod.aggregate_reportability(
            [self._rec("partition_00001", status="failed"),
             self._rec("partition_00002", official=True)],
            official_test_label="partition_00002",
            final_test_is_latest_available=True,
        )
        self.assertIn("partition_00001", agg["failed_partition_labels_before_official_test"])

    # ---- official_test_block (failed official -> first-class block, P1.3) ----

    def test_official_block_reports_failed_official(self) -> None:
        records = [{
            "partition": "2026-06-10_to_2026-06-13", "ordinal": 1, "is_official_latest_test": True,
            "status": "failed", "exception_type": "ValueError", "error": "ValueError('bad')", "skipped": True,
        }]
        block = self.mod.official_test_block(records, True)
        self.assertIsNotNone(block)
        self.assertEqual(block["status"], "failed")
        self.assertFalse(block["reportable"])
        self.assertEqual(block["exception_type"], "ValueError")

    # ---- failure classifier (P0.1) ----

    def test_fatal_partition_failure_classifier(self) -> None:
        fatal = self.mod.is_fatal_partition_failure
        # data-contract ValueErrors are skippable (not fatal)
        self.assertFalse(fatal(ValueError("label_valid_mask must be a subset of decision action validity.")))
        # bug-shaped exceptions are fatal
        self.assertTrue(fatal(TypeError("x")))
        self.assertTrue(fatal(AttributeError("x")))
        self.assertTrue(fatal(KeyError("x")))
        self.assertTrue(fatal(MemoryError()))
        # resource/CUDA failures are fatal even if raised as RuntimeError/ValueError
        self.assertTrue(fatal(RuntimeError("CUDA out of memory. Tried to allocate ...")))
        self.assertTrue(fatal(ValueError("CUDA error: device-side assert triggered")))

    def test_skip_flag_defaults_off(self) -> None:
        self.assertFalse(self.mod.parse_args(["--run-name", "x"]).skip_failed_partitions)
        self.assertTrue(self.mod.parse_args(["--run-name", "x", "--skip-failed-partitions"]).skip_failed_partitions)


if __name__ == "__main__":
    unittest.main()
