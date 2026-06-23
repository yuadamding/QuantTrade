"""Regression tests pinning the repo-wide review fixes (2026-06).

Each test locks a specific defect the review found so it cannot silently return:
  * #1  the train.second CLI flag/attr/preset mismatch that made `qt train second` non-runnable
  * #6  the SSL forward-market target hardcoding CASH at column 0
  * #18 EvaluationProtocol.validate never checking train_start
  * #22 validate_action_mask accepting float 1.0/0.0; per-row issue cap overshoot
  * #4  the point-in-time leakage guard failing OPEN for a masked-valid bar with no timestamp
  * #5  the Stage-2 decision-policy trainer silently ignoring an unsupported dynamic-state knob
"""

from __future__ import annotations

import unittest

import torch

from _support import ROOT, load_script


class CliFlagRegressionTests(unittest.TestCase):
    TRAINER_SCRIPTS = (
        "train_hour_from_second_rl",
        "train_hourly_from_second_protocol_partitions",
        "train_hourly_from_second_calendar_holdout",
    )

    def test_train_second_preset_parses_and_resolves_second_layers(self) -> None:
        from rl_quant.workflows.presets import resolve_preset

        module = load_script("train_hour_from_second_rl")
        args = module.parse_args(resolve_preset("train.second"))
        # The preset forwards --second-layers; the script must accept it AND expose args.second_layers (the
        # SecondToHourTrainingConfig field). The old --minute-layers flag/attr must be gone.
        self.assertEqual(args.second_layers, 2)
        self.assertFalse(hasattr(args, "minute_layers"))

    def test_all_second_trainers_use_second_layers_flag(self) -> None:
        for name in self.TRAINER_SCRIPTS:
            text = (ROOT / "scripts" / f"{name}.py").read_text()
            self.assertIn("--second-layers", text, f"{name}: must define --second-layers")
            self.assertNotIn("--minute-layers", text, f"{name}: stale --minute-layers flag must be gone")
            self.assertIn("args.second_layers", text, f"{name}: config must read args.second_layers")


class SslCashIndexTests(unittest.TestCase):
    def test_forward_market_targets_excludes_the_real_cash_column(self) -> None:
        from rl_quant.training.context_pretrain import forward_market_targets

        # CASH at column 1 (not 0). Columns 0 and 2 are real risky actions.
        action_returns = torch.tensor([[0.10, 0.0, 0.30], [float("nan"), 0.0, -0.02]])
        valid = torch.tensor([[True, True, True], [False, True, True]])

        target = forward_market_targets(action_returns, valid, cash_index=1)
        self.assertEqual(tuple(target.shape), (2, 2))
        self.assertAlmostEqual(float(target[0, 0]), 0.20, places=5)   # mean(0.10, 0.30), CASH col 1 excluded
        self.assertAlmostEqual(float(target[1, 0]), -0.02, places=5)  # only the valid non-CASH return

        # The default (cash_index=0) excludes a DIFFERENT column, proving the parameter is honored.
        default_target = forward_market_targets(action_returns, valid)
        self.assertAlmostEqual(float(default_target[0, 0]), 0.15, places=5)  # mean(0.0, 0.30), col 0 excluded


class EvaluationProtocolTrainStartTests(unittest.TestCase):
    def _proto(self, **overrides):
        from rl_quant.evaluation.research_protocol import EvaluationProtocol

        base = dict(
            name="x",
            train_start=None,
            train_end="2026-02-01T00:00:00+00:00",
            val_end="2026-03-01T00:00:00+00:00",
            test_start="2026-04-01T00:00:00+00:00",
            test_end=None,
            benchmark_names=["CASH"],
        )
        base.update(overrides)
        return EvaluationProtocol(**base)

    def test_train_start_after_train_end_is_rejected(self) -> None:
        from rl_quant.evaluation.research_protocol import ResearchProtocolError

        with self.assertRaises(ResearchProtocolError):
            self._proto(train_start="2026-03-15T00:00:00+00:00").validate()

    def test_tz_naive_train_start_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            self._proto(train_start="2026-01-01T00:00:00").validate()

    def test_valid_train_start_passes(self) -> None:
        self._proto(train_start="2026-01-01T00:00:00+00:00").validate()  # must not raise


class ActionMaskValidatorTests(unittest.TestCase):
    def test_float_mask_entries_are_rejected(self) -> None:
        from rl_quant.protocol.validators import validate_action_mask

        ok, issues = validate_action_mask([[1.0, 0.0]])
        self.assertFalse(ok)
        self.assertTrue(any("non-boolean" in i for i in issues))

    def test_int_and_bool_masks_pass(self) -> None:
        from rl_quant.protocol.validators import validate_action_mask

        self.assertTrue(validate_action_mask([[1, 0]])[0])
        self.assertTrue(validate_action_mask([[True, False]])[0])

    def test_per_row_issue_cap_is_not_overshot_on_a_wide_row(self) -> None:
        from rl_quant.protocol.validators import validate_invalid_returns_are_nan

        # One row, 200 actions, all marked VALID but all NaN -> 200 candidate issues. The cap must bound
        # the returned list to ~max_issues, not the full row width.
        action_returns = [[float("nan")] * 200]
        valid_mask = [[True] * 200]
        ok, issues = validate_invalid_returns_are_nan(action_returns, valid_mask, max_issues=5)
        self.assertFalse(ok)
        self.assertLessEqual(len(issues), 5 + 1)  # capped issues + the truncation sentinel


class SecondTimestampLeakageGuardTests(unittest.TestCase):
    def _payload(self, second_ts, valid):
        return {
            "decision_timestamps": ["2026-06-12T11:00:00+00:00"],
            "next_timestamps": ["2026-06-12T12:00:00+00:00"],
            # 1 row, 1 hour, 1 second. source_bar_interval defaults to non-1s so no latency floor applies.
            "second_timestamp_grid": [[[second_ts]]],
            "second_mask": torch.tensor([[[valid]]], dtype=torch.bool),
        }

    def test_masked_valid_bar_with_empty_timestamp_raises(self) -> None:
        from rl_quant.datasets.hour_from_second import validate_second_timestamp_grid

        with self.assertRaisesRegex(ValueError, "no timestamp"):
            validate_second_timestamp_grid(self._payload("", True))

    def test_masked_invalid_bar_with_empty_timestamp_is_skipped(self) -> None:
        from rl_quant.datasets.hour_from_second import validate_second_timestamp_grid

        validate_second_timestamp_grid(self._payload("", False))  # must not raise

    def test_valid_bar_available_before_decision_passes(self) -> None:
        from rl_quant.datasets.hour_from_second import validate_second_timestamp_grid

        validate_second_timestamp_grid(self._payload("2026-06-12T10:30:00+00:00", True))  # ts <= decision


class Stage2GuardTests(unittest.TestCase):
    def test_dynamic_feature_dim_is_rejected_by_stage2_trainer(self) -> None:
        from test_second_to_hour import SecondToHourTests

        from rl_quant.training.context_pretrain import ContextPretrainConfig, train_second_context_encoder
        from rl_quant.training.decision_policy import (
            DecisionPolicyConfig,
            precompute_context_embeddings,
            train_decision_policy_dqn,
        )

        data = SecondToHourTests._small_second_to_hour_split()
        encoder, _head, _ = train_second_context_encoder(
            data,
            ContextPretrainConfig(epochs=1, batch_size=2, d_model=16, n_heads=2, second_layers=1,
                                  hour_layers=1, feedforward_dim=16, max_second_tokens=None),
        )
        embeddings = precompute_context_embeddings(encoder, data)
        with self.assertRaisesRegex(ValueError, "dynamic_feature_dim"):
            train_decision_policy_dqn(
                embeddings, data,
                DecisionPolicyConfig(d_model=16, dynamic_feature_dim=4, num_envs=2, episode_length=2,
                                     train_steps=2, batch_size=2, warmup_steps=2),
            )


class NewsLlmAggregateGoldenTests(unittest.TestCase):
    def test_aggregate_schema_hash_and_count_are_pinned(self) -> None:
        from rl_quant.features.news_llm import (
            NEWS_LLM_AGGREGATE_FEATURE_NAMES,
            NEWS_LLM_AGGREGATE_SCHEMA_HASH,
        )

        # The 28 features are POSITIONAL; a reorder/rename silently mislabels a model input. This golden hash
        # (and the persisted action_news_llm_schema_hash derived from it) changes on any such edit.
        self.assertEqual(len(NEWS_LLM_AGGREGATE_FEATURE_NAMES), 28)
        self.assertEqual(
            NEWS_LLM_AGGREGATE_SCHEMA_HASH,
            "cc6b907803b6a05c3cea98df76d4c0e85c8f80406d1ec1a118b7ab7596ec3166",
        )

    def test_aggregate_values_catch_field_and_window_swaps(self) -> None:
        from rl_quant.features.news_llm import (
            DAY_MS,
            HOUR_MS,
            NEWS_LLM_AGGREGATE_FEATURE_NAMES,
            aggregate_news_llm_features_for_symbol,
        )

        decision_ms = 100 * DAY_MS

        def row(article_id, published_ms, **kw):
            base = dict(article_id=article_id, published_timestamp_ms=published_ms,
                        llm_feature_available_timestamp_ms=published_ms, llm_valid=True, article_weight=1.0)
            base.update(kw)
            return base

        rows = [
            row("A", decision_ms - 2 * HOUR_MS, positive_score=0.8, negative_score=0.0, sentiment_score=0.4),
            row("B", decision_ms - 3 * DAY_MS, positive_score=0.0, negative_score=0.0, sentiment_score=0.9),
        ]
        values, *_ = aggregate_news_llm_features_for_symbol(
            rows=rows, decision_ms=decision_ms, source_available=True
        )
        at = NEWS_LLM_AGGREGATE_FEATURE_NAMES.index
        # FIELD swap (positive<->negative): 1d window = article A only (B is 3 days old).
        self.assertAlmostEqual(values[at("llm_positive_intensity_1d")], 0.8, places=5)
        self.assertAlmostEqual(values[at("llm_negative_intensity_1d")], 0.0, places=5)
        # WINDOW swap (1d<->7d): net sentiment is A only in 1d (0.4) but mean(A,B) in 7d (0.65).
        self.assertAlmostEqual(values[at("llm_net_sentiment_1d")], 0.4, places=5)
        self.assertAlmostEqual(values[at("llm_net_sentiment_7d")], 0.65, places=5)


class EnvDayBoundaryParityTests(unittest.TestCase):
    def _multi_day_split(self):
        from rl_quant.datasets.hour_from_second import HourFromMinuteDataSplit

        ts = [
            "2026-01-02T14:30:00+00:00", "2026-01-02T15:30:00+00:00",
            "2026-01-05T14:30:00+00:00", "2026-01-05T15:30:00+00:00",
        ]
        nxt = [
            "2026-01-02T15:30:00+00:00", "2026-01-02T16:30:00+00:00",
            "2026-01-05T15:30:00+00:00", "2026-01-05T16:30:00+00:00",
        ]
        n = len(ts)
        return HourFromMinuteDataSplit(
            name="t", decision_timestamps=ts, next_timestamps=nxt,
            second_feature_names=["m"], hour_feature_names=["h"], action_names=["CASH", "QQQ"],
            second_features=torch.zeros((n, 1, 1, 1), dtype=torch.float32),
            second_mask=torch.ones((n, 1, 1), dtype=torch.bool),
            hour_features=torch.zeros((n, 1, 1), dtype=torch.float32),
            action_returns=torch.zeros((n, 2), dtype=torch.float32),
            valid_start_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            valid_index_mask=torch.tensor([True, True, True, True]),
            second_feature_mean=torch.zeros(1), second_feature_std=torch.ones(1),
            hour_feature_mean=torch.zeros(1), hour_feature_std=torch.ones(1),
            hours_lookback=1, seconds_per_hour=1,
        )

    def test_precomputed_day_id_reproduces_date_prefix_equivalence(self) -> None:
        # #11 byte-identity: the precomputed per-row day id must induce EXACTLY the same equivalence classes
        # as the old decision_timestamps[i][:10] string compare (same prefix <-> same id), so the vectorized
        # day-boundary reset matches the prior O(num_envs) Python-loop reset.
        from rl_quant.envs.second_to_hour import SecondToHourEnvConfig, VectorizedSecondToHourEnv

        data = self._multi_day_split()
        env = VectorizedSecondToHourEnv(
            data, SecondToHourEnvConfig(num_envs=2, episode_length=2), torch.device("cpu")
        )
        day_id = env._day_id.tolist()
        ts = data.decision_timestamps
        self.assertEqual(len(day_id), len(ts))
        for i in range(len(ts)):
            for j in range(len(ts)):
                self.assertEqual(
                    day_id[i] == day_id[j], ts[i][:10] == ts[j][:10],
                    f"day-id parity broke at ({i}, {j})",
                )


if __name__ == "__main__":
    unittest.main()
