"""Enforces that QuantTrade's learning framework follows the specified design.

The point of Phase-1 is to verify the package implements the design, so these are the design as executable
assertions:
  1. CONTEXT/POLICY SPLIT is structural -- the context encoder has no policy concept, and (the literal test)
     after the encoder is frozen and used to encode embeddings, a policy training step leaves NO gradient on
     the encoder (the policy gradient cannot reach the context).
  2. CONTEXT IS CAUSAL -- a stock's context (its most-recent-valid token output) is invariant to perturbing
     later/padded tokens, so information only rolls forward from the session open.
  3. POLICY IS A PERMUTATION-EQUIVARIANT SET over actions producing an allocation -- weights are a simplex over
     {CASH, stocks}, unavailable actions get ~0 weight, CASH is always allocatable, and the SAME head runs for
     51 or hundreds of actions (shared weights => scales).
"""
from __future__ import annotations

import unittest

import torch

from rl_quant.datasets import CHUNK_FEATS, COV_FIELDS, NEWS_FEATS
from rl_quant.models import (
    ContextEncoder,
    ContextEncoderConfig,
    DecisionPolicyConfig,
    DecisionPolicyHead,
)
from rl_quant.training import encode_windows, evaluate_policy, freeze_encoder, train_decision_policy

D, A, C = 3, 6, 5
NC, NN = len(COV_FIELDS), NEWS_FEATS


def _encoder(d_model=16):
    return ContextEncoder(ContextEncoderConfig(chunk_feature_dim=CHUNK_FEATS, d_model=d_model, n_heads=2,
                                               n_layers=1, feedforward_dim=32, dropout=0.0, max_chunks=C))


def _policy(d_model=16):
    return DecisionPolicyHead(DecisionPolicyConfig(context_dim=d_model, covariate_dim=NC, news_dim=NN,
                                                   token_dim=16, n_heads=2, n_layers=1, feedforward_dim=32))


def _synthetic_window(seed=0, decisions=D, actions=A, chunks=C):
    g = torch.Generator().manual_seed(seed)
    chunk = torch.randn(decisions, actions, chunks, CHUNK_FEATS, generator=g)
    mask = torch.zeros(decisions, actions, chunks, dtype=torch.bool)
    for di in range(decisions):
        for ai in range(1, actions):  # action 0 = CASH carries no stock context
            k = 1 + (di + ai) % chunks  # left-aligned valid prefix of length k
            mask[di, ai, :k] = True
    ret = 0.01 * torch.randn(decisions, actions, generator=g)
    ret[:, 0] = 0.0
    valid = mask.any(-1)
    valid[:, 0] = True
    return {"chunk": chunk, "chunk_mask": mask, "cov": torch.randn(decisions, actions, NC, generator=g),
            "news": torch.randn(decisions, actions, NN, generator=g), "ret": ret, "ret_valid": valid,
            "decisions": decisions}


class ContextIsPolicyFree(unittest.TestCase):
    def test_encoder_constructor_and_forward_have_no_policy_concept(self):
        enc = _encoder()
        # no policy attributes on the context encoder
        for banned in ("action", "policy", "previous", "constraint", "cash", "score"):
            self.assertFalse(any(banned in n.lower() for n, _ in enc.named_parameters()),
                             f"context encoder must not carry a '{banned}' parameter")
        per_stock, market = enc(torch.randn(2, A, C, CHUNK_FEATS), torch.ones(2, A, C, dtype=torch.bool))
        self.assertEqual(per_stock.shape, (2, A, enc.d_model))
        self.assertEqual(market.shape, (2, enc.d_model))

    def test_policy_gradient_cannot_reach_the_frozen_encoder(self):
        torch.manual_seed(0)
        enc = _encoder()
        freeze_encoder(enc)
        self.assertTrue(all(not p.requires_grad for p in enc.parameters()))
        emb = encode_windows(enc, [_synthetic_window(1)], torch.device("cpu"), max_decisions=D)
        self.assertFalse(emb[0]["per_stock"].requires_grad, "cached context must be detached")
        train_decision_policy(_policy(), emb, steps=1, eval_every=0, val_emb=emb,
                              device=torch.device("cpu"), batch_windows=1)
        self.assertTrue(all(p.grad is None for p in enc.parameters()),
                        "policy training put a gradient on the context encoder -> the split is broken")


class ContextIsCausal(unittest.TestCase):
    def test_context_is_invariant_to_future_and_padded_tokens(self):
        torch.manual_seed(0)
        enc = _encoder()
        enc.eval()
        chunk = torch.randn(1, 1, C, CHUNK_FEATS)
        mask = torch.zeros(1, 1, C, dtype=torch.bool)
        mask[0, 0, :2] = True  # only the first two tokens are valid (left-aligned)
        with torch.no_grad():
            base, _ = enc(chunk, mask)
            perturbed = chunk.clone()
            perturbed[0, 0, 2:] += 5.0  # change ONLY the later/padded tokens
            after, _ = enc(perturbed, mask)
        self.assertTrue(torch.allclose(base, after, atol=1e-5),
                        "context changed when only future/padded tokens changed -> not causal")


class PolicyIsAllocationOverActions(unittest.TestCase):
    def _weights(self, actions):
        torch.manual_seed(0)
        pol = _policy()
        avail = torch.ones(1, actions, dtype=torch.bool)
        avail[0, actions - 1] = False  # one unavailable action
        prev = torch.zeros(1, actions)
        prev[0, 0] = 1.0
        return pol(torch.randn(1, 16), torch.randn(1, actions, 16), torch.randn(1, actions, NC),
                   torch.randn(1, actions, NN), prev, avail)

    def test_weights_form_a_simplex_and_respect_constraints(self):
        w = self._weights(A)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertGreaterEqual(float(w.min()), 0.0)
        self.assertLess(float(w[0, A - 1]), 1e-6, "unavailable action received weight")
        self.assertGreater(float(w[0, 0]), 0.0, "CASH must remain allocatable")

    def test_same_head_scales_to_many_actions(self):
        for actions in (51, 256):
            w = self._weights(actions)
            self.assertEqual(w.shape, (1, actions))
            self.assertAlmostEqual(float(w.sum()), 1.0, places=4)


class EndToEndStagesRun(unittest.TestCase):
    def test_encode_then_policy_eval_produces_per_decision_returns(self):
        torch.manual_seed(0)
        enc = _encoder()
        freeze_encoder(enc)
        emb = encode_windows(enc, [_synthetic_window(2), _synthetic_window(3)], torch.device("cpu"), max_decisions=D)
        train_decision_policy(_policy(), emb, steps=2, eval_every=0, val_emb=emb,
                              device=torch.device("cpu"), batch_windows=2)
        rows = evaluate_policy(_policy(), emb, torch.device("cpu"), cost=5e-4)
        self.assertEqual(len(rows), 2 * D)  # one net return per real decision, no padding
        self.assertTrue(all(abs(r) < 1.0 for r in rows))


if __name__ == "__main__":
    unittest.main()
