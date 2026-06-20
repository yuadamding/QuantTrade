from __future__ import annotations

import unittest
from rl_quant.research_protocol import (
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
from _support import load_script


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

    def test_dataset_manifest_records_return_basis_and_tolerates_superset_keys(self) -> None:
        # The manifest now carries the canonical action-return basis (so the reportability agreement check has a
        # declared-side basis), and from_dict tolerates the builder's SUPERSET extras (reportable / missing_*)
        # by filtering to declared fields rather than raising.
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
            action_return_weight_semantics="full_capital_single_slot_returns",
            action_return_formula="clipped_simple_return(entry_fill, exit_fill)",
            action_return_clip_min=-1.0,
            action_return_clip_max=1.0,
            action_return_semantics_version="v1",
            action_return_fill_convention="first_close_at_or_after_decision_plus_execution_latency",
        )
        manifest.validate()  # basis fields do not affect validate()
        payload = manifest.to_dict()
        self.assertEqual(payload["action_return_weight_semantics"], "full_capital_single_slot_returns")
        self.assertEqual(payload["action_return_fill_convention"],
                         "first_close_at_or_after_decision_plus_execution_latency")
        # Round-trips, and from_dict drops the builder's non-field extras (reportable / reportability_errors /
        # missing_*) instead of raising on them.
        enriched = {**payload, "reportable": True, "reportability_errors": [], "missing_action_source_symbols": []}
        restored = DatasetManifest.from_dict(enriched)
        self.assertEqual(restored, manifest)

        # A TYPO of a protected action_return_* basis key is REJECTED (not silently dropped): a dropped typo
        # would vacuum the manifest-side basis and the agreement check.
        with self.assertRaisesRegex(ResearchProtocolError, "action_return_"):
            DatasetManifest.from_dict({**payload, "action_return_fill_conventon": "x"})

        # The basis lands where the reportability agreement check reads it (action_return_* keys via ReturnBasis).
        from rl_quant.datasets.hour_from_subhour import ReturnBasis
        self.assertTrue(ReturnBasis.from_mapping(payload).is_complete())

    def test_dataset_manifest_basis_field_names_match_return_basis_reader(self) -> None:
        # Cross-module invariant: the manifest's action_return_* field names MUST equal the payload keys that
        # ReturnBasis reads, or the declared-side basis silently goes all-None (agreement vacuous) on a rename.
        from dataclasses import fields as dc_fields

        from rl_quant.datasets.hour_from_subhour import _RETURN_BASIS_FIELD_KEYS

        manifest_basis_fields = {f.name for f in dc_fields(DatasetManifest) if f.name.startswith("action_return_")}
        self.assertEqual(manifest_basis_fields, set(_RETURN_BASIS_FIELD_KEYS.values()))

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
            selection_split="validation",
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
            selection_split="validation",
            baseline_results=[BaselineResult("CASH", 0.0, None, 0.0)],
            cost_stress_results=[StressTestResult("2x_cost", "cost", "multiplier", 2.0, 0.0, None, 0.0)],
            frequency_stress_results=[
                StressTestResult("min_hold_2", "frequency", "min_hold_bars", 2.0, 0.0, None, 0.0)
            ],
        )

    def test_model_manifest_rejects_test_split_selection(self) -> None:
        manifest = self._reportable_manifest()
        manifest.validate_reportable()  # selection_split="validation" -> ok
        # The central design claim: selection_split is the gate, selected_by is descriptive-only. In strict mode a
        # selected_by that mentions "test" must NOT trip the gate as long as selection_split=="validation" (the
        # free-text label is genuinely ignored -- the leakage check is never inferred from it under strict).
        manifest.selected_by = "best_on_test_set_metric"
        manifest.validate_reportable()  # selection_split="validation" still ok despite "test" in selected_by
        manifest.selected_by = "validation_net_return"
        # Structured anti-leakage gate: a missing or non-validation selection_split fails strict reportability,
        # regardless of the free-text selected_by label (no longer the enforced field, and never inferred from).
        manifest.selection_split = None
        with self.assertRaisesRegex(ResearchProtocolError, "selection_split"):
            manifest.validate_reportable()
        manifest.selection_split = "test"
        with self.assertRaisesRegex(ResearchProtocolError, "selection_split"):
            manifest.validate_reportable()
        # Legacy compatibility (strict=False) falls back to the brittle selected_by text heuristic.
        manifest.selection_split = None
        manifest.selected_by = "test_total_return"
        with self.assertRaisesRegex(ResearchProtocolError, "selected_by must reference validation"):
            manifest.validate_reportable(strict=False)
        manifest.selected_by = "validation_net_return"
        manifest.validate_reportable(strict=False)  # legacy: 'validation' in the label -> ok

    def test_model_manifest_strict_requires_nonempty_identity_fields(self) -> None:
        # Strict reportability requires the identity/provenance fields to be non-empty (not just present).
        for field_name in ("created_at_utc", "algorithm", "encoder", "training_dataset_id",
                            "feature_names_hash", "action_names_hash"):
            manifest = self._reportable_manifest()
            setattr(manifest, field_name, "")
            with self.assertRaisesRegex(ResearchProtocolError, field_name):
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
