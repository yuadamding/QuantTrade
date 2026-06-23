"""Context-learning / decision-policy split: the encoder is policy-free, trained self-supervised; the policy
scores actions from a (frozen) context embedding. See models.second_to_hour / models.decision_policy."""

from __future__ import annotations

import inspect
import math
import unittest

import torch

from rl_quant.models.decision_policy import DecisionPolicyQNetwork
from rl_quant.models.second_to_hour import SecondToHourContextEncoder
from rl_quant.protocol.constraints import CONSTRAINT_FEATURE_DIM
from rl_quant.training.context_pretrain import (
    ContextPretrainConfig,
    encode_split,
    forward_market_targets,
    train_second_context_encoder,
)
from rl_quant.training.decision_policy import precompute_context_embeddings

from test_second_to_hour import SecondToHourTests  # reuse the proven tiny-split builder


class ContextPolicySplitTests(unittest.TestCase):
    def test_context_encoder_is_policy_free(self) -> None:
        src = inspect.getsource(SecondToHourContextEncoder)
        for token in ("previous_action", "constraint", "action_feature", "transition_", "dynamic_", "self.head"):
            self.assertNotIn(token, src, f"context encoder must be policy-free; found {token!r}")
        params = set(inspect.signature(SecondToHourContextEncoder.__init__).parameters)
        for bad in ("action_count", "action_embedding_dim", "constraint_feature_dim", "action_feature_dim",
                    "transition_feature_dim", "dynamic_feature_dim"):
            self.assertNotIn(bad, params, f"context encoder ctor must not take policy arg {bad!r}")

    def test_context_encoder_forward_shape(self) -> None:
        enc = SecondToHourContextEncoder(
            second_feature_dim=3, hour_feature_dim=2, hours_lookback=2, seconds_per_hour=4,
            d_model=16, n_heads=2, second_layers=1, hour_layers=1, feedforward_dim=16,
        )
        ctx = enc(torch.randn(5, 2, 4, 3), torch.ones(5, 2, 4, dtype=torch.bool), torch.randn(5, 2, 2))
        self.assertEqual(tuple(ctx.shape), (5, 16))

    def test_context_encoder_finite_with_leading_masked_seconds(self) -> None:
        """Regression: an hour whose LEADING seconds are masked (data starts late) must not poison the
        embedding with NaN. Under the causal mask, query positions before the first valid second would
        attend to an all-padding window -> softmax over all -inf -> NaN, propagating to the read-out
        token. The encoder must keep the embedding finite (guarantee key 0 is always attendable)."""
        torch.manual_seed(0)
        enc = SecondToHourContextEncoder(
            second_feature_dim=3, hour_feature_dim=2, hours_lookback=2, seconds_per_hour=6,
            d_model=16, n_heads=2, second_layers=2, hour_layers=1, feedforward_dim=16,
            max_second_tokens=None,  # no compression: exercise the raw causal+padding interaction
        )
        enc.eval()
        second = torch.randn(3, 2, 6, 3)
        mask = torch.ones(3, 2, 6, dtype=torch.bool)
        mask[1, 0, :4] = False   # row 1, hour 0: first 4 seconds masked, last 2 valid (leading padding)
        mask[2, 1, :] = False    # row 2, hour 1: fully empty (the previously-handled case)
        ctx = enc(second, mask, torch.randn(3, 2, 2))
        self.assertEqual(tuple(ctx.shape), (3, 16))
        self.assertTrue(bool(torch.isfinite(ctx).all()), "leading-masked seconds must not yield NaN context")

    def test_policy_scores_from_context_embedding(self) -> None:
        policy = DecisionPolicyQNetwork(d_model=16, action_count=4)
        q = policy(torch.randn(5, 16), torch.zeros(5, dtype=torch.long), torch.zeros(5, CONSTRAINT_FEATURE_DIM))
        self.assertEqual(tuple(q.shape), (5, 4))

    def test_forward_market_targets_masks_cash_and_invalid(self) -> None:
        action_returns = torch.tensor([[0.0, 0.01, 0.03], [0.0, float("nan"), -0.02]])
        valid = torch.tensor([[True, True, True], [True, False, True]])
        target = forward_market_targets(action_returns, valid)
        self.assertEqual(tuple(target.shape), (2, 2))
        self.assertAlmostEqual(float(target[0, 0]), 0.02, places=5)   # mean(0.01, 0.03), CASH excluded
        self.assertAlmostEqual(float(target[1, 0]), -0.02, places=5)  # only the valid non-CASH return

    def test_ssl_pretrain_runs_and_encode_split(self) -> None:
        data = SecondToHourTests._small_second_to_hour_split()
        encoder, _head, metrics = train_second_context_encoder(
            data,
            ContextPretrainConfig(epochs=1, batch_size=2, d_model=16, n_heads=2, second_layers=1,
                                  hour_layers=1, feedforward_dim=16, max_second_tokens=None),
        )
        self.assertTrue(math.isfinite(metrics["last_epoch_loss"]))
        embeddings = encode_split(encoder, data)
        self.assertEqual(embeddings.shape[0], int(data.second_features.shape[0]))
        self.assertEqual(embeddings.shape[1], 16)
        precompute_context_embeddings(encoder, data)  # freezes the encoder
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))

    def test_stage2_decision_policy_dqn_over_embeddings(self) -> None:
        import math

        import torch

        from rl_quant.training.decision_policy import (
            DecisionPolicyConfig,
            precompute_context_embeddings,
            train_decision_policy_dqn,
        )

        torch.manual_seed(0)
        data = SecondToHourTests._small_second_to_hour_split()
        encoder, _head, _ = train_second_context_encoder(
            data,
            ContextPretrainConfig(epochs=1, batch_size=2, d_model=16, n_heads=2, second_layers=1,
                                  hour_layers=1, feedforward_dim=16, max_second_tokens=None),
        )
        embeddings = precompute_context_embeddings(encoder, data)
        policy, metrics = train_decision_policy_dqn(
            embeddings, data,
            DecisionPolicyConfig(d_model=16, num_envs=4, episode_length=4, train_steps=40,
                                 batch_size=8, warmup_steps=8, target_update_interval=10),
        )
        self.assertEqual(metrics["train_steps"], 40)
        self.assertTrue(metrics["final_loss"] is not None and math.isfinite(metrics["final_loss"]))
        rows = int(data.action_returns.shape[0])
        q = policy(embeddings, torch.zeros(rows, dtype=torch.long), torch.zeros(rows, CONSTRAINT_FEATURE_DIM))
        self.assertEqual(tuple(q.shape), (rows, len(data.action_names)))
        self.assertTrue(bool(torch.isfinite(q).all()))

    def test_stage2_best_validation_selection_returns_best_checkpoint(self) -> None:
        """Review #4: with the frozen encoder + a val split, Stage-2 must select the best-VALIDATION
        checkpoint (not the final one) and report the selection metrics."""
        import math

        import torch

        from rl_quant.training.decision_policy import (
            DecisionPolicyConfig,
            precompute_context_embeddings,
            train_decision_policy_dqn,
        )

        torch.manual_seed(0)
        data = SecondToHourTests._small_second_to_hour_split()
        encoder, _head, _ = train_second_context_encoder(
            data,
            ContextPretrainConfig(epochs=1, batch_size=2, d_model=16, n_heads=2, second_layers=1,
                                  hour_layers=1, feedforward_dim=16, max_second_tokens=None),
        )
        embeddings = precompute_context_embeddings(encoder, data)
        policy, metrics = train_decision_policy_dqn(
            embeddings, data,
            DecisionPolicyConfig(d_model=16, num_envs=4, episode_length=4, train_steps=40,
                                 batch_size=8, warmup_steps=8, target_update_interval=10, eval_interval=10),
            encoder=encoder, val_data=data,
        )
        self.assertTrue(metrics["val_selected"])
        self.assertIsNotNone(metrics["best_val_return"])
        self.assertTrue(math.isfinite(metrics["best_val_return"]))
        self.assertIsNotNone(metrics["best_val_step"])
        # The val trace was populated and the encoder stayed frozen (selection must not train Stage-1).
        self.assertGreaterEqual(len(metrics["val_trace"]), 1)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
