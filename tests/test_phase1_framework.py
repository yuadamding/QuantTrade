"""Enforces that QuantTrade's learning framework follows the specified EVENT-TIMED design.

The point of Phase-1 is to verify the package implements the design, so these are the design as executable
assertions:
  1. CONTEXT/POLICY SPLIT is structural -- the context encoder has no policy concept, and (the literal test)
     after the encoder is frozen and used to encode per-block embeddings, a policy training step leaves NO
     gradient on the encoder (the policy gradient cannot reach the context).
  2. CONTEXT IS CAUSAL ACROSS BLOCKS -- a block's context is invariant to perturbing LATER blocks' seconds, so
     information only rolls forward from the session open (two-tier: local within blocks, causal over blocks).
  3. POLICY IS A PERMUTATION-EQUIVARIANT SET over actions producing an allocation + an ACT-GATE -- weights are a
     simplex over {CASH, stocks}, unavailable actions get ~0 weight, CASH is always allocatable, the SAME head
     runs for 51 or hundreds of actions (shared weights => scales), and the per-block gate is in [0,1] (WHEN to
     trade), trained under a soft per-day budget with T+1 execution.
"""
from __future__ import annotations

import unittest

import torch

from rl_quant.datasets import BAR_FEATS, COV_FIELDS, NEWS_RAW_DIM
from rl_quant.models import (
    ContextEncoder,
    ContextEncoderConfig,
    ContextForwardHead,
    DecisionPolicyConfig,
    DecisionPolicyHead,
    PerStockForwardHead,
)
from rl_quant.training import (
    encode_days,
    evaluate_policy,
    freeze_encoder,
    policy_telemetry,
    ssl_targets,
    ssl_targets_perstock,
    train_context_encoder,
    train_decision_policy,
)

A, S, BL, M = 6, 16, 4, 4           # actions, raw-second tokens (session), block_seconds, news articles
NB = S // BL                         # blocks per session (4)
NC, NRD = len(COV_FIELDS), NEWS_RAW_DIM


def _encoder(d_model=16, max_seconds=S, block_seconds=BL, n_layers=2):
    return ContextEncoder(ContextEncoderConfig(bar_feature_dim=BAR_FEATS, covariate_dim=NC, d_model=d_model,
                                               n_heads=2, n_layers=n_layers, feedforward_dim=32, dropout=0.0,
                                               max_seconds=max_seconds, block_seconds=block_seconds))


def _policy(d_model=16, **kw):
    return DecisionPolicyHead(DecisionPolicyConfig(context_dim=d_model, news_raw_dim=NRD, max_news=M, token_dim=16,
                                                   n_heads=2, n_layers=1, feedforward_dim=32, **kw))


def _news(b, actions, gen=None):  # raw per-article scores + mask (what the policy aggregates in-model)
    return (torch.randn(b, actions, M, NRD, generator=gen), torch.ones(b, actions, M, dtype=torch.bool))


def _synthetic_day(seed=0, actions=A, seconds=S, block_seconds=BL):
    """One trading day: a full session of RAW bars + per-block covariates/news/T+1 labels (the training unit).
    T+1 labels exist for blocks 0..nB-3 (the last two blocks have no forward label), mirroring build_window."""
    g = torch.Generator().manual_seed(seed)
    nb = seconds // block_seconds
    bars = torch.randn(actions, seconds, BAR_FEATS, generator=g)  # RAW bars (one token/second)
    mask = torch.zeros(actions, seconds, dtype=torch.bool)
    mask[1:] = True                                              # action 0 = CASH carries no stock context
    ret = 0.01 * torch.randn(nb, actions, generator=g)
    ret[:, 0] = 0.0
    valid = torch.zeros(nb, actions, dtype=torch.bool)
    valid[:, 0] = True                                          # CASH valid (return 0) at every block
    valid[: max(0, nb - 2), 1:] = True                          # stocks: T+1 label for blocks 0..nb-3
    return {"bars": bars, "bar_mask": mask,
            "cov_blocks": torch.randn(nb, actions, NC, generator=g),
            "news_raw": torch.randn(nb, actions, M, NRD, generator=g),
            "news_mask": torch.ones(nb, actions, M, dtype=torch.bool),
            "ret": ret, "ret_valid": valid}


class ContextIsPolicyFree(unittest.TestCase):
    def test_encoder_constructor_and_forward_have_no_policy_concept(self):
        enc = _encoder()
        for banned in ("action", "policy", "previous", "constraint", "cash", "score", "gate"):
            self.assertFalse(any(banned in n.lower() for n, _ in enc.named_parameters()),
                             f"context encoder must not carry a '{banned}' parameter")
        enc.eval()
        per_stock, market = enc(torch.randn(2, A, S, BAR_FEATS), torch.ones(2, A, S, dtype=torch.bool),
                                torch.randn(2, NB, A, NC))
        self.assertEqual(per_stock.shape, (2, NB, A, enc.d_model))   # a context at EVERY block
        self.assertEqual(market.shape, (2, NB, enc.d_model))

    def test_policy_gradient_cannot_reach_the_frozen_encoder(self):
        torch.manual_seed(0)
        enc = _encoder()
        freeze_encoder(enc)
        self.assertTrue(all(not p.requires_grad for p in enc.parameters()))
        emb = encode_days(enc, [_synthetic_day(1)], torch.device("cpu"), batch=1)
        self.assertFalse(emb[0]["per_stock"].requires_grad, "cached context must be detached")
        train_decision_policy(_policy(), emb, steps=1, eval_every=0, val_days=emb,
                              device=torch.device("cpu"), batch_days=1)
        self.assertTrue(all(p.grad is None for p in enc.parameters()),
                        "policy training put a gradient on the context encoder -> the split is broken")


class ContextIsCausal(unittest.TestCase):
    def test_single_block_context_invariant_to_future_and_padded_tokens(self):
        torch.manual_seed(0)
        enc = _encoder(block_seconds=S, max_seconds=S)            # one block -> tier-1 only
        enc.eval()
        bars = torch.randn(1, 1, S, BAR_FEATS)
        mask = torch.zeros(1, 1, S, dtype=torch.bool)
        mask[0, 0, :2] = True                                    # only the first two seconds are valid
        cov = torch.randn(1, 1, A, NC)[:, :, :1]                 # cov for the single (CASH-less) stock
        with torch.no_grad():
            base, _ = enc(bars, mask, cov)
            perturbed = bars.clone()
            perturbed[0, 0, 2:] += 5.0                           # change ONLY later/padded second-tokens
            after, _ = enc(perturbed, mask, cov)
        self.assertTrue(torch.allclose(base, after, atol=1e-5),
                        "context changed when only future/padded tokens changed -> not causal")

    def test_two_tier_earlier_blocks_invariant_to_later_blocks(self):
        torch.manual_seed(0)
        enc = _encoder(d_model=16, max_seconds=S, block_seconds=BL, n_layers=4)   # NB=4 blocks of 4 seconds
        enc.eval()
        bars = torch.randn(1, A, S, BAR_FEATS)
        mask = torch.ones(1, A, S, dtype=torch.bool)             # all seconds valid
        cov = torch.randn(1, NB, A, NC)
        with torch.no_grad():
            ps, market = enc(bars, mask, cov)
            self.assertEqual(ps.shape, (1, NB, A, 16))
            self.assertEqual(market.shape, (1, NB, 16))
            perturbed = bars.clone()
            perturbed[0, :, 2 * BL:] += 9.0                      # change only blocks 2,3 (seconds 8..15)
            ps2, _ = enc(perturbed, mask, cov)
        self.assertTrue(torch.allclose(ps[:, :2], ps2[:, :2], atol=1e-5),
                        "an earlier block's context changed when only LATER blocks changed -> not causal")


class PolicyIsAllocationAndGate(unittest.TestCase):
    def _forward(self, actions):
        torch.manual_seed(0)
        pol = _policy()
        avail = torch.ones(1, actions, dtype=torch.bool)
        avail[0, actions - 1] = False                            # one unavailable action
        prev = torch.zeros(1, actions)
        prev[0, 0] = 1.0
        ns, nm = _news(1, actions)
        return pol(torch.randn(1, 16), torch.randn(1, actions, 16), ns, nm, prev, avail)

    def test_weights_form_a_simplex_and_respect_constraints(self):
        w, gate = self._forward(A)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertGreaterEqual(float(w.min()), 0.0)
        self.assertLess(float(w[0, A - 1]), 1e-6, "unavailable action received weight")
        self.assertGreater(float(w[0, 0]), 0.0, "CASH must remain allocatable")
        self.assertTrue(0.0 <= float(gate) <= 1.0, "act-gate must be a probability in [0,1]")
        self.assertEqual(gate.shape, (1,))

    def test_same_head_scales_to_many_actions(self):
        for actions in (51, 256):
            w, gate = self._forward(actions)
            self.assertEqual(w.shape, (1, actions))
            self.assertAlmostEqual(float(w.sum()), 1.0, places=4)
            self.assertTrue(0.0 <= float(gate) <= 1.0)


class DesignSeriesIsValid(unittest.TestCase):
    def test_designs_load_and_are_internally_consistent(self):
        from rl_quant.training import DEFAULT_DESIGN, DESIGNS, SWEEP
        self.assertIn(DEFAULT_DESIGN, DESIGNS)
        self.assertTrue(set(SWEEP).issubset(DESIGNS))
        for name, d in DESIGNS.items():
            self.assertEqual(d.d_model % d.enc_heads, 0, f"{name}: enc_heads must divide d_model")
            self.assertEqual(d.policy_token_dim % d.policy_heads, 0, f"{name}: policy_heads must divide token_dim")
            self.assertEqual(d.session_seconds % d.block_seconds, 0, f"{name}: block_seconds must divide session")
            self.assertGreater(d.max_actions_per_day, 0)
            self.assertGreaterEqual(d.budget_lambda, 0)
            self.assertGreater(min(d.ssl_steps, d.policy_steps, d.ssl_batch_size, d.batch_days), 0)

    def test_a_design_drives_the_models_at_its_settings(self):
        from rl_quant.training import DESIGNS
        d = DESIGNS["wide"]
        enc = ContextEncoder(ContextEncoderConfig(bar_feature_dim=BAR_FEATS, covariate_dim=NC, d_model=d.d_model,
                             n_heads=d.enc_heads, n_layers=d.enc_layers, feedforward_dim=d.d_model * 4,
                             dropout=d.dropout, max_seconds=S, block_seconds=BL))
        enc.eval()
        per_stock, market = enc(torch.randn(1, A, S, BAR_FEATS), torch.ones(1, A, S, dtype=torch.bool),
                                torch.randn(1, NB, A, NC))
        self.assertEqual(market.shape, (1, NB, d.d_model))
        pol = DecisionPolicyHead(DecisionPolicyConfig(context_dim=d.d_model, news_raw_dim=NRD, max_news=M,
                                 token_dim=d.policy_token_dim, n_heads=d.policy_heads, n_layers=d.policy_layers,
                                 feedforward_dim=d.policy_token_dim * 2))
        ns, nm = _news(1, A)
        w, gate = pol(market[:, 0], per_stock[:, 0], ns, nm, torch.zeros(1, A), torch.ones(1, A, dtype=torch.bool))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=4)
        self.assertTrue(0.0 <= float(gate) <= 1.0)


class TrainingStrategyKnobs(unittest.TestCase):
    def test_lr_schedule_warmup_then_decay(self):
        from rl_quant.training._optim import lr_scale
        self.assertAlmostEqual(lr_scale(0, 100, 10), 0.1, places=6)      # warmup start
        self.assertAlmostEqual(lr_scale(9, 100, 10), 1.0, places=6)      # warmup end
        self.assertLess(lr_scale(99, 100, 10), 0.05)                     # cosine ~ -> 0
        self.assertEqual(lr_scale(50, 100, 10, "constant"), 1.0)         # constant = flat

    def test_temperature_sharpens_or_diversifies_allocation(self):
        torch.manual_seed(0)
        ms, ps = torch.randn(1, 16), torch.randn(1, A, 16)
        prev, avail = torch.zeros(1, A), torch.ones(1, A, dtype=torch.bool)
        ns, nm = _news(1, A)
        torch.manual_seed(1)
        w_sharp, _ = _policy(temperature=0.3)(ms, ps, ns, nm, prev, avail)
        torch.manual_seed(1)
        w_diff, _ = _policy(temperature=3.0)(ms, ps, ns, nm, prev, avail)

        def ent(w):
            return float(-(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum())
        self.assertLess(ent(w_sharp), ent(w_diff))   # low temp -> concentrated (lower entropy)

    def test_ssl_context_training_step_runs_and_targets_are_per_block(self):
        torch.manual_seed(0)
        enc, head = _encoder(), ContextForwardHead(16)
        days = [_synthetic_day(5), _synthetic_day(6)]
        self.assertEqual(ssl_targets(days[0]["ret"], days[0]["ret_valid"]).shape, (NB, 2))
        train_context_encoder(enc, head, days, device=torch.device("cpu"), steps=2, batch_size=2, accum_steps=1,
                              warmup_steps=1, schedule="constant")

    def test_amp_entropy_and_budget_training_step_runs(self):
        torch.manual_seed(0)
        enc = _encoder()
        freeze_encoder(enc)
        emb = encode_days(enc, [_synthetic_day(2), _synthetic_day(3)], torch.device("cpu"), batch=2)
        # high budget penalty with a tiny per-day cap exercises the gate-budget term; must stay finite
        train_decision_policy(_policy(), emb, steps=2, eval_every=0, val_days=emb, device=torch.device("cpu"),
                              batch_days=2, entropy_coef=0.02, grad_clip=0.5, warmup_steps=1, schedule="constant",
                              max_actions=0.0, budget_lambda=10.0)


class EndToEndStagesRun(unittest.TestCase):
    def test_encode_then_policy_eval_produces_per_decision_returns(self):
        torch.manual_seed(0)
        enc = _encoder()
        freeze_encoder(enc)
        days = [_synthetic_day(2), _synthetic_day(3)]
        emb = encode_days(enc, days, torch.device("cpu"), batch=2)
        train_decision_policy(_policy(), emb, steps=2, eval_every=0, val_days=emb,
                              device=torch.device("cpu"), batch_days=2)
        rows = evaluate_policy(_policy(), emb, torch.device("cpu"), cost=5e-4)
        self.assertEqual(len(rows), len(days) * (NB - 2))  # one net per label-valid block (last 2 blocks unlabeled)
        self.assertTrue(all(abs(r) < 1.0 for r in rows))


def _planted_days(n_days=10, actions=A, nb=8, d=16, seed=0):
    """Synthetic per-block embeddings where each stock's context LINEARLY encodes its next-block cross-sectionally
    -demeaned return (a strong planted relative-value signal). A working policy must keep the gate OPEN and tilt
    toward the high-signal stocks; the all-CASH collapse would score ~0. Bypasses the encoder to isolate Stage 2."""
    g = torch.Generator().manual_seed(seed)
    days = []
    for _ in range(n_days):
        per_stock = torch.randn(nb, actions, d, generator=g)
        ret = torch.zeros(nb, actions)
        ret[:, 1:] = 0.02 * torch.tanh(per_stock[:, 1:, 0])          # planted: context dim 0 -> next-block return
        ret[:, 1:] = ret[:, 1:] - ret[:, 1:].mean(1, keepdim=True)   # cross-sectionally demeaned (pure relative)
        valid = torch.zeros(nb, actions, dtype=torch.bool)
        valid[:, 0] = True
        valid[: nb - 2, 1:] = True                                   # last 2 blocks unlabeled (T+1)
        days.append({"market": per_stock.mean(1), "per_stock": per_stock,
                     "news_raw": torch.zeros(nb, actions, M, NRD), "news_mask": torch.zeros(nb, actions, M, dtype=torch.bool),
                     "ret": ret, "ret_valid": valid, "n_blocks": nb})
    return days


class PolicyLearnsAndTradesUnderFix(unittest.TestCase):
    """Locks the CASH-collapse fix: the gate starts open and the policy learns a planted cross-sectional signal
    (it trades and is profitable) rather than abstaining to ~0 -- the failure mode of the unnormalized budget."""

    def test_gate_initialized_open(self):
        torch.manual_seed(0)
        pol = _policy()                                              # gate_init_bias defaults to 2.0
        _, gate = pol(torch.randn(4, 16), torch.randn(4, A, 16), *_news(4, A),
                      torch.zeros(4, A), torch.ones(4, A, dtype=torch.bool))
        self.assertGreater(float(gate.mean()), 0.5, "act-gate must START OPEN (trade), not closed")

    def test_policy_learns_planted_signal_and_keeps_trading(self):
        torch.manual_seed(0)
        days = _planted_days()
        pol = _policy()
        dev = torch.device("cpu")
        r0 = evaluate_policy(pol, days, dev, cost=1e-4)
        before = sum(r0) / len(r0)
        train_decision_policy(pol, days, steps=250, lr=3e-3, batch_days=10, cost=1e-4, risk_lambda=0.0,
                              max_actions=5.0, budget_lambda=0.1, gate_entropy_coef=1e-3,
                              friction_warmup_steps=60, schedule="constant", eval_every=0, val_days=days, device=dev)
        rows = evaluate_policy(pol, days, dev, cost=1e-4)
        oos = sum(rows) / len(rows)
        tele = policy_telemetry(pol, days, dev, cost=1e-4)
        self.assertGreater(tele["mean_gate"], 0.3, "gate COLLAPSED to ~hold -> the CASH-basin bug is back")
        self.assertGreater(oos, 0.0, "policy did not learn the planted cross-sectional signal")
        self.assertGreater(oos, before, "training did not improve OOS over the untrained policy")


class SSLPerStockPretext(unittest.TestCase):
    def test_perstock_target_is_demeaned_and_cash_is_invalid(self):
        day = _synthetic_day(7)
        tgt, mask = ssl_targets_perstock(day["ret"], day["ret_valid"])
        self.assertEqual(tgt.shape, (NB, A))
        self.assertFalse(bool(mask[:, 0].any()), "CASH must be an invalid per-stock target")
        for b in range(NB):                                          # demeaned -> valid non-CASH targets sum ~0
            m = mask[b]
            if m.sum() >= 2:
                self.assertAlmostEqual(float(tgt[b][m].sum()), 0.0, places=5)

    def test_perstock_ssl_head_trains_and_receives_gradient(self):
        torch.manual_seed(0)
        enc, head, ps_head = _encoder(), ContextForwardHead(16), PerStockForwardHead(16)
        days = [_synthetic_day(5), _synthetic_day(6)]
        train_context_encoder(enc, head, days, device=torch.device("cpu"), perstock_head=ps_head, perstock_coef=1.0,
                              steps=2, batch_size=2, accum_steps=1, warmup_steps=1, schedule="constant")
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps_head.parameters()),
                        "per-stock SSL head received no gradient -> the cross-sectional pretext is not wired")


if __name__ == "__main__":
    unittest.main()
