"""Focused model-contract tests for the soft 30-session holding policy."""

from __future__ import annotations

import unittest

import torch

from rl_quant.models.daily_policy import (
    HOLD30_AGE_CAP,
    HOLD30_ALPHA_MODEL_SETTING_IDS,
    HOLD30_MODEL_SETTING_IDS,
    HOLD30_V2_MODEL_SETTING_IDS,
    DailyCrossSectionConfig,
    DailyCrossSectionPolicy,
    clip_hold30_hazard_residual,
    exact_hold30_intent,
    hold30_proposed_release,
    hold30_release_hazard,
    resolve_hold30_model_switches,
)


def _config(setting_id: str | None = None) -> DailyCrossSectionConfig:
    return DailyCrossSectionConfig(
        context_dim=4,
        bar_feature_dim=5,
        raw_policy_dim=4,
        raw_policy_layers=2,
        raw_policy_heads=1,
        raw_block_seconds=2,
        session_seconds=4,
        news_raw_dim=1,
        max_news=2,
        news_embed_dim=4,
        token_dim=8,
        temporal_layers=1,
        temporal_heads=1,
        daily_lookback=4,
        max_days=8,
        alloc_layers=1,
        alloc_heads=1,
        feedforward_dim=16,
        dropout=0.0,
        hold30_setting=setting_id,
    )


class Hold30HazardContract(unittest.TestCase):
    def test_finite_endpoints_and_boundary_gradients(self) -> None:
        raw = torch.tensor([-13.0, -12.0, -11.0, 0.0, 11.0, 12.0, 13.0], requires_grad=True)
        clipped = clip_hold30_hazard_residual(raw)
        torch.testing.assert_close(
            clipped,
            torch.tensor([-12.0, -12.0, -11.0, 0.0, 11.0, 12.0, 12.0]),
        )
        clipped.sum().backward()
        torch.testing.assert_close(raw.grad, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))

        ages = torch.arange(HOLD30_AGE_CAP + 1, dtype=torch.float64)
        hold = hold30_release_hazard(ages, torch.tensor(-12.0, dtype=torch.float64))
        maximum = hold30_release_hazard(ages, torch.tensor(12.0, dtype=torch.float64))
        self.assertTrue(torch.equal(hold, torch.zeros_like(hold)))
        self.assertTrue(bool((maximum > 0.0).all()))
        self.assertTrue(bool((maximum <= 1.0).all()))

    def test_zero_residual_prior_has_documented_release_clock(self) -> None:
        # New close-t purchases first face the hazard after earning one return,
        # so the first evaluated post-return age is one.
        survival = torch.tensor(1.0, dtype=torch.float64)
        expected_duration = torch.tensor(0.0, dtype=torch.float64)
        cdf = torch.tensor(0.0, dtype=torch.float64)
        median = None
        for session in range(1, 512):
            age = torch.tensor(min(session, HOLD30_AGE_CAP), dtype=torch.float64)
            q = hold30_release_hazard(age, torch.tensor(0.0, dtype=torch.float64))
            sale_probability = survival * q
            expected_duration = expected_duration + session * sale_probability
            cdf = cdf + sale_probability
            survival = survival * (1.0 - q)
            if median is None and float(cdf) >= 0.5:
                median = session
            if float(survival) < 1e-15:
                break
        expected_duration = expected_duration + session * survival
        self.assertAlmostEqual(float(expected_duration), 30.4092035936, places=9)
        self.assertEqual(median, 31)

    def test_exact_neutral_intent_releases_no_cohort_mass(self) -> None:
        reference = torch.tensor([[0.10, 0.35, 0.55], [0.20, 0.30, 0.50]], dtype=torch.float64)
        intent = exact_hold30_intent(reference)
        self.assertIsNotNone(intent.entry_scores)
        self.assertIsNotNone(intent.hazard_residual)
        self.assertIsNotNone(intent.exposure_residual)
        torch.testing.assert_close(intent.entry_scores, torch.zeros_like(reference))
        torch.testing.assert_close(intent.hazard_residual, torch.full_like(reference, -12.0))
        torch.testing.assert_close(intent.exposure_residual, torch.zeros(2, dtype=torch.float64))

        cohorts = torch.rand(2, 3, HOLD30_AGE_CAP + 1, dtype=torch.float64)
        released = hold30_proposed_release(cohorts, intent.hazard_residual)
        self.assertTrue(torch.equal(released, torch.zeros_like(released)))


class Hold30ActorContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.state = torch.randn(2, 5, 8)
        self.previous = torch.tensor(
            [[0.10, 0.25, 0.20, 0.30, 0.15], [0.20, 0.10, 0.25, 0.15, 0.30]]
        )
        self.available = torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        )
        self.age = torch.rand(2, 5, 5)

    def test_legacy_eight_setting_switches_and_intent_shapes(self) -> None:
        self.assertEqual(len(HOLD30_V2_MODEL_SETTING_IDS), 8)
        self.assertEqual(len(HOLD30_ALPHA_MODEL_SETTING_IDS), 8)
        self.assertEqual(HOLD30_MODEL_SETTING_IDS, HOLD30_V2_MODEL_SETTING_IDS)
        self.assertTrue(set(HOLD30_V2_MODEL_SETTING_IDS).isdisjoint(HOLD30_ALPHA_MODEL_SETTING_IDS))
        for setting_id in HOLD30_V2_MODEL_SETTING_IDS:
            with self.subTest(setting_id=setting_id):
                torch.manual_seed(19)
                policy = DailyCrossSectionPolicy(_config(setting_id)).eval()
                switches = resolve_hold30_model_switches(setting_id)
                age = self.age if switches.use_age_input else None
                intent = policy.hold30_intent(self.state, self.previous, self.available, age)
                if switches.mechanism in ("H0", "H1"):
                    self.assertEqual(intent.target_logits.shape, (2, 5))
                    self.assertEqual(intent.gate.shape, (2,))
                    self.assertIsNone(intent.entry_scores)
                    self.assertIsNone(intent.hazard_residual)
                    self.assertIsNone(intent.exposure_residual)
                elif switches.mechanism == "H3":
                    self.assertEqual(intent.entry_scores.shape, (2, 5))
                    self.assertIsNone(intent.target_logits)
                    self.assertIsNone(intent.gate)
                    self.assertIsNone(intent.hazard_residual)
                    self.assertIsNone(intent.exposure_residual)
                else:
                    self.assertEqual(intent.entry_scores.shape, (2, 5))
                    self.assertEqual(intent.hazard_residual.shape, (2, 5))
                    self.assertEqual(intent.exposure_residual.shape, (2,))
                    self.assertEqual(float(intent.entry_scores[0, 4].detach()), 0.0)
                    self.assertTrue(torch.equal(intent.hazard_residual[:, 0], torch.full((2,), -12.0)))
                    self.assertEqual(float(intent.hazard_residual[0, 4].detach()), -12.0)
                    self.assertEqual(intent.exposure_residual.dtype, torch.float32)

    def test_h2_entry_is_market_only_but_hazard_consumes_position_state(self) -> None:
        policy = DailyCrossSectionPolicy(_config("hold30-m02-age-hazard")).eval()
        first = policy.hold30_intent(self.state, self.previous, self.available, self.age)
        changed_previous = torch.flip(self.previous, dims=(1,))
        changed_age = 1.0 - self.age
        second = policy.hold30_intent(self.state, changed_previous, self.available, changed_age)
        torch.testing.assert_close(first.entry_scores, second.entry_scores, rtol=0.0, atol=0.0)
        self.assertGreater(float((first.hazard_residual - second.hazard_residual).abs().max().detach()), 0.0)

    def test_no_age_and_no_exposure_ablation_semantics(self) -> None:
        no_age = DailyCrossSectionPolicy(_config("hold30-a04-no-age-input")).eval()
        without_age = no_age.hold30_intent(self.state, self.previous, self.available)
        with_ignored_age = no_age.hold30_intent(self.state, self.previous, self.available, self.age * 99.0)
        torch.testing.assert_close(without_age.entry_scores, with_ignored_age.entry_scores, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            without_age.hazard_residual, with_ignored_age.hazard_residual, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            without_age.exposure_residual, with_ignored_age.exposure_residual, rtol=0.0, atol=0.0
        )

        no_exposure = DailyCrossSectionPolicy(_config("hold30-a07-no-exp-timing")).eval()
        intent = no_exposure.hold30_intent(self.state, self.previous, self.available, self.age)
        self.assertTrue(torch.equal(intent.exposure_residual, torch.zeros(2)))
        self.assertIsNone(no_exposure.exposure_head)

    def test_gate_initialization_matches_h0_and_h1_contracts(self) -> None:
        h0 = DailyCrossSectionPolicy(_config("hold30-m00-legacy-gate")).eval()
        h1 = DailyCrossSectionPolicy(_config("hold30-m01-slow-gate")).eval()
        h0_gate = h0.hold30_intent(self.state, self.previous, self.available).gate
        h1_gate = h1.hold30_intent(self.state, self.previous, self.available).gate
        torch.testing.assert_close(h0_gate, torch.full((2,), torch.sigmoid(torch.tensor(2.0))), atol=2e-3, rtol=0.0)
        torch.testing.assert_close(
            h1_gate,
            torch.full((2,), 1.0 - torch.exp(torch.tensor(-1.0 / 30.0))),
            atol=2e-3,
            rtol=0.0,
        )

    def test_legacy_config_requires_no_age_and_retains_step_api(self) -> None:
        policy = DailyCrossSectionPolicy(_config()).eval()
        weights, gate = policy.step(self.state, self.previous, self.available)
        self.assertEqual(weights.shape, (2, 5))
        self.assertEqual(gate.shape, (2,))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0.0)
        with self.assertRaisesRegex(RuntimeError, "hold30_setting"):
            policy.hold30_intent(self.state, self.previous, self.available)


if __name__ == "__main__":
    unittest.main()
