from __future__ import annotations

import unittest
import torch
from rl_quant.decision_framework import (
    ActionEligibility,
    DataQualityReport,
    DecisionDataset,
    DecisionFrameworkError,
    DecisionLog,
    DecisionSnapshot,
    FeatureManifest,
    ReadinessConfig,
    action_eligibilities_to_mask,
    apply_data_quality_gate,
    assert_available_at,
    decision_readiness_score,
    filter_point_in_time_rows,
    readiness_band,
    validate_reportable_summary,
)


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
