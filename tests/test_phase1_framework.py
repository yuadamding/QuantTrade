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

from rl_quant.datasets import BAR_FEATS, COV_FIELDS, NEWS_RAW_DIM, build_daily_episodes, cross_day_returns
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
    evaluate_policy_detailed,
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
    return DecisionPolicyHead(DecisionPolicyConfig(context_dim=d_model, bar_feature_dim=BAR_FEATS,
                                                   raw_policy_dim=4, raw_block_seconds=BL,
                                                   raw_policy_layers=0, raw_policy_heads=1,
                                                   news_raw_dim=NRD, max_news=M, token_dim=16,
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
    avail = torch.ones(nb, actions, dtype=torch.bool)            # as-of tradeability (all present in this synthetic)
    return {"bars": bars, "bar_mask": mask,
            "cov_blocks": torch.randn(nb, actions, NC, generator=g),
            "news_raw": torch.randn(nb, actions, M, NRD, generator=g),
            "news_mask": torch.ones(nb, actions, M, dtype=torch.bool),
            "avail": avail, "ret": ret, "ret_valid": valid}


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
        pol = _policy()
        train_decision_policy(pol, emb, steps=1, eval_every=0, val_days=emb,
                              device=torch.device("cpu"), batch_days=1)
        self.assertTrue(all(p.grad is None for p in enc.parameters()),
                        "policy training put a gradient on the context encoder -> the split is broken")
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol.raw_encoder.parameters()),
                        "policy raw-second encoder received no profit-gradient signal")


class ContextIsCausal(unittest.TestCase):
    def test_context_encoder_runs_with_padding_mask(self):
        torch.manual_seed(0)
        enc = _encoder()
        enc.eval()
        bars = torch.randn(2, A, S, BAR_FEATS)
        mask = torch.ones(2, A, S, dtype=torch.bool)
        mask[:, 0, :] = False                                    # fully masked CASH row exercises the fallback
        mask[:, 1:, -3:] = False                                 # partially padded stock rows exercise SDPA masks
        cov = torch.randn(2, NB, A, NC)
        with torch.no_grad():
            per_stock, market = enc(bars, mask, cov)
        self.assertEqual(per_stock.shape, (2, NB, A, enc.d_model))
        self.assertEqual(market.shape, (2, NB, enc.d_model))
        self.assertTrue(torch.isfinite(per_stock).all())
        self.assertTrue(torch.isfinite(market).all())

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
        raw = torch.randn(1, actions, 4)
        return pol(torch.randn(1, 16), torch.randn(1, actions, 16), raw, ns, nm, prev, avail)

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
        pol = DecisionPolicyHead(DecisionPolicyConfig(context_dim=d.d_model, bar_feature_dim=BAR_FEATS,
                                 raw_policy_dim=d.raw_policy_dim, raw_block_seconds=BL,
                                 raw_policy_layers=d.raw_policy_layers, raw_policy_heads=d.raw_policy_heads,
                                 news_raw_dim=NRD, max_news=M,
                                 token_dim=d.policy_token_dim, n_heads=d.policy_heads, n_layers=d.policy_layers,
                                 feedforward_dim=d.policy_token_dim * 2))
        ns, nm = _news(1, A)
        raw = torch.randn(1, A, d.raw_policy_dim)
        w, gate = pol(market[:, 0], per_stock[:, 0], raw, ns, nm, torch.zeros(1, A),
                      torch.ones(1, A, dtype=torch.bool))
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
        raw = torch.randn(1, A, 4)
        torch.manual_seed(1)
        w_sharp, _ = _policy(temperature=0.3)(ms, ps, raw, ns, nm, prev, avail)
        torch.manual_seed(1)
        w_diff, _ = _policy(temperature=3.0)(ms, ps, raw, ns, nm, prev, avail)

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


class MissingLabelsAreNotFlatReturns(unittest.TestCase):
    def test_evaluation_excludes_allocations_to_missing_label_actions(self):
        class PickMissing(torch.nn.Module):
            def encode_raw_policy_step(self, bars, bar_mask, step):
                return torch.zeros(bars.shape[0], bars.shape[1], 1)

            def forward(self, market, per_stock, raw_policy_ctx, news_scores, news_mask, prev_weights, available):
                w = torch.zeros_like(prev_weights)
                w[:, 1] = 1.0
                return w, torch.ones(prev_weights.shape[0])

        day = {"market": torch.zeros(1, 1), "per_stock": torch.zeros(1, 3, 1),
               "bars": torch.zeros(3, BL, BAR_FEATS), "bar_mask": torch.ones(3, BL, dtype=torch.bool),
               "news_raw": torch.zeros(1, 3, M, NRD), "news_mask": torch.zeros(1, 3, M, dtype=torch.bool),
               "avail": torch.ones(1, 3, dtype=torch.bool),
               "ret": torch.tensor([[0.0, float("nan"), 0.05]]),
               "ret_valid": torch.tensor([[True, False, True]]), "n_blocks": 1}
        self.assertEqual(evaluate_policy(PickMissing(), [day], torch.device("cpu"), cost=0.0), [])
        rows, stats = evaluate_policy_detailed(PickMissing(), [day], torch.device("cpu"), cost=0.0)
        self.assertEqual(rows, [])
        self.assertEqual(stats["total_blocks"], 1)
        self.assertEqual(stats["label_blocks"], 1)
        self.assertEqual(stats["reportable_blocks"], 0)
        self.assertEqual(stats["reportable_fraction"], 0.0)
        self.assertEqual(stats["label_reportable_fraction"], 0.0)
        self.assertAlmostEqual(stats["mean_missing_label_weight"], 1.0, places=5)
        tele = policy_telemetry(PickMissing(), [day], torch.device("cpu"), cost=0.0)
        self.assertAlmostEqual(tele["mean_missing_label_weight"], 1.0, places=5)


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
                     "bars": torch.zeros(actions, nb * BL, BAR_FEATS),
                     "bar_mask": torch.ones(actions, nb * BL, dtype=torch.bool),
                     "news_raw": torch.zeros(nb, actions, M, NRD),
                     "news_mask": torch.zeros(nb, actions, M, dtype=torch.bool),
                     "avail": torch.ones(nb, actions, dtype=torch.bool),
                     "ret": ret, "ret_valid": valid, "n_blocks": nb})
    return days


class PolicyLearnsAndTradesUnderFix(unittest.TestCase):
    """Locks the CASH-collapse fix: the gate starts open and the policy learns a planted cross-sectional signal
    (it trades and is profitable) rather than abstaining to ~0 -- the failure mode of the unnormalized budget."""

    def test_gate_initialized_open(self):
        torch.manual_seed(0)
        pol = _policy()                                              # gate_init_bias defaults to 2.0
        _, gate = pol(torch.randn(4, 16), torch.randn(4, A, 16), torch.randn(4, A, 4), *_news(4, A),
                      torch.zeros(4, A), torch.ones(4, A, dtype=torch.bool))
        self.assertGreater(float(gate.mean()), 0.5, "act-gate must START OPEN (trade), not closed")

    def test_policy_learns_planted_signal_and_keeps_trading(self):
        torch.manual_seed(0)
        days = _planted_days()
        pol = _policy()
        dev = torch.device("cpu")
        r0 = evaluate_policy(pol, days, dev, cost=1e-4)
        before = sum(r0) / len(r0)
        train_decision_policy(pol, days, steps=140, lr=5e-3, batch_days=10, cost=1e-4, risk_lambda=0.0,
                              max_actions=5.0, budget_lambda=0.1, gate_entropy_coef=1e-3,
                              friction_warmup_steps=30, schedule="constant", eval_every=0, val_days=days, device=dev)
        rows = evaluate_policy(pol, days, dev, cost=1e-4)
        oos = sum(rows) / len(rows)
        tele = policy_telemetry(pol, days, dev, cost=1e-4)
        self.assertGreater(tele["mean_gate"], 0.3, "gate COLLAPSED to ~hold -> the CASH-basin bug is back")
        self.assertGreater(oos, 0.0, "policy did not learn the planted cross-sectional signal")
        self.assertGreater(oos, before, "training did not improve OOS over the untrained policy")

    def test_tight_budget_sparsifies_trading_for_long_holds(self):
        # A tight per-episode trade budget must drive the gate SPARSE so positions are HELD across the episode
        # (the mechanism behind "two trades >=180 days apart"): over a 20-step episode the gate starts ~0.88
        # (>=17 trades) and must drop well below that under budget pressure.
        torch.manual_seed(0)
        days = _planted_days(n_days=6, nb=20)
        pol = _policy()
        dev = torch.device("cpu")
        train_decision_policy(pol, days, steps=140, lr=5e-3, batch_days=6, cost=1e-4, risk_lambda=0.0,
                              max_actions=2.0, budget_lambda=2.0, gate_entropy_coef=0.0, friction_warmup_steps=0,
                              schedule="constant", eval_every=0, val_days=days, device=dev)
        tele = policy_telemetry(pol, days, dev, cost=1e-4)
        self.assertLess(tele["mean_gate"], 0.5, "tight budget did not sparsify the gate -> long holds not enabled")
        self.assertLess(tele["trades_per_day"], 10.0, "still trading most days despite a 2-trade/episode budget")


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


def _daily_records(n=12, actions=A, d=16, seed=0):
    """A date-sorted sequence of per-day end-of-day-context records (what the daily assembler consumes)."""
    g = torch.Generator().manual_seed(seed)
    return [{"date": f"2022-02-{i + 1:02d}", "day_open": 1.0 + 0.5 * torch.rand(actions, generator=g),
             "avail": torch.ones(actions, dtype=torch.bool),
             "market": torch.randn(d, generator=g), "per_stock": torch.randn(actions, d, generator=g),
             "bars": torch.zeros(actions, S, BAR_FEATS), "bar_mask": torch.ones(actions, S, dtype=torch.bool),
             "news_raw": torch.zeros(actions, M, NRD), "news_mask": torch.zeros(actions, M, dtype=torch.bool)}
            for i in range(n)]


class CrossDayDailyMode(unittest.TestCase):
    def test_cross_day_returns_are_open_to_open_T1(self):
        # CASH col 0 (dummy), stock1 opens 1.0,1.1,1.21,..., stock2 2.0,2.2,2.42,...
        op = torch.tensor([[1., 1.00, 2.00], [1., 1.10, 2.20], [1., 1.21, 2.42],
                           [1., 1.00, 1.00], [1., 1.00, 1.00]])
        ret, valid = cross_day_returns(op)
        self.assertAlmostEqual(float(ret[0, 1]), 1.21 / 1.10 - 1.0, places=5)  # open[d+2]/open[d+1]-1
        self.assertEqual(float(ret[0, 0]), 0.0)                                # CASH return 0
        self.assertTrue(bool(valid[0, 0]) and bool(valid[0, 1]))
        self.assertFalse(bool(valid[3, 1]) or bool(valid[4, 1]))               # last 2 days: no T+1 label

    def test_build_daily_episodes_shape_and_trains(self):
        recs = _daily_records(12)
        eps = build_daily_episodes(recs, episode_len=4)
        self.assertEqual(len(eps), 2)                                          # usable=10 -> floor over 4-day chunks
        self.assertEqual(eps[0]["per_stock"].shape, (4, A, 16))
        self.assertEqual(eps[0]["market"].shape, (4, 16))
        self.assertEqual(eps[0]["ret"].shape, (4, A))
        dev = torch.device("cpu")
        train_decision_policy(_policy(), eps, steps=2, batch_days=2, val_days=eps, device=dev, eval_every=0,
                              budget_lambda=0.0)
        rows = evaluate_policy(_policy(), eps, dev, cost=1e-4)
        self.assertEqual(len(rows), 2 * 4)                                     # every day label-valid (opens>0)
        self.assertTrue(all(abs(r) < 1.0 for r in rows))

    def test_truncated_bptt_enables_multiday_credit_assignment(self):
        # Delayed-reward task: the winner is identifiable ONLY from the day-0 context; its payoff lands on the
        # HELD days 1-2 (day-0 return ~0, later contexts uninformative). A myopic (1-step) policy can never
        # connect the day-0 decision to the later payoff -> ~0; truncated BPTT credits the held position's
        # multi-day returns to the day-0 decision -> it learns to buy & hold the winner. This is the mechanism
        # behind learning long-range holds.
        def _delayed(n_ep=8, L=4, actions=4, d=16, seed=0):
            gg = torch.Generator().manual_seed(seed)
            eps = []
            for e in range(n_ep):
                win = 1 + (e % (actions - 1))
                ps = 0.1 * torch.randn(L, actions, d, generator=gg)     # later days: uninformative context
                ps[0, :, 0] = -1.0
                ps[0, win, 0] = 2.0                                     # day-0 signal ONLY identifies the winner
                ret = torch.zeros(L, actions)
                ret[1, win] = 0.03
                ret[2, win] = 0.03                                      # payoff on the HELD days (not day 0)
                ret[:, 1:] = ret[:, 1:] - ret[:, 1:].mean(1, keepdim=True)
                valid = torch.zeros(L, actions, dtype=torch.bool)
                valid[:, 0] = True
                valid[:3, 1:] = True
                eps.append({"market": ps.mean(1), "per_stock": ps,
                            "bars": torch.zeros(L, actions, S, BAR_FEATS),
                            "bar_mask": torch.ones(L, actions, S, dtype=torch.bool),
                            "news_raw": torch.zeros(L, actions, M, NRD),
                            "news_mask": torch.zeros(L, actions, M, dtype=torch.bool),
                            "avail": torch.ones(L, actions, dtype=torch.bool),
                            "ret": ret, "ret_valid": valid, "n_blocks": L})
            return eps

        dev = torch.device("cpu")

        def train_eval(window):
            torch.manual_seed(0)
            pol = _policy()
            train_decision_policy(pol, _delayed(seed=1), steps=180, lr=5e-3, batch_days=8, cost=1e-3,
                                  risk_lambda=0.0, budget_lambda=0.0, gate_entropy_coef=0.0, friction_warmup_steps=0,
                                  schedule="constant", eval_every=0, val_days=_delayed(seed=1), device=dev,
                                  bptt_window=window)
            r = evaluate_policy(pol, _delayed(seed=1), dev, cost=1e-3)
            return sum(r) / len(r)

        oos_myopic = train_eval(1)
        oos_bptt = train_eval(4)
        self.assertGreater(oos_bptt, oos_myopic, "BPTT did not improve multi-day credit over the myopic baseline")
        self.assertGreater(oos_bptt, 0.0, "BPTT policy failed to capture the delayed (held) reward")

    def test_long_hold_overlapping_windows_and_short_split(self):
        # overlapping sliding windows -> many fixed-length episodes from a long sequence (LONG-HOLD training data)
        eps = build_daily_episodes(_daily_records(30), episode_len=10, stride=5)
        self.assertGreater(len(eps), 2)
        self.assertTrue(all(e["per_stock"].shape[0] == 10 for e in eps))
        # a held position (gate=0) carries across the WHOLE episode -> a 10-day hold here, >=180 at full scale
        # short split (fewer days than episode_len) is NOT starved: one episode of the full usable length
        short = build_daily_episodes(_daily_records(8), episode_len=180)
        self.assertEqual(len(short), 1)
        self.assertEqual(short[0]["per_stock"].shape[0], 6)                    # usable = 8 - 2 (T+1 tail)


def _fake_window(dates, actions=3, nb=4, seconds=8):
    """A minimal built-window dict (real shapes, zero content) for testing the leak-critical split logic."""
    Dd = len(dates)
    return {"bars": torch.zeros(Dd, actions, seconds, BAR_FEATS), "bar_mask": torch.zeros(Dd, actions, seconds, dtype=torch.bool),
            "cov_blocks": torch.zeros(Dd, nb, actions, NC), "news_raw": torch.zeros(Dd, nb, actions, M, NRD),
            "news_mask": torch.zeros(Dd, nb, actions, M, dtype=torch.bool), "avail": torch.ones(Dd, nb, actions, dtype=torch.bool),
            "ret": torch.zeros(Dd, nb, actions), "ret_valid": torch.zeros(Dd, nb, actions, dtype=torch.bool),
            "day_open": torch.ones(Dd, actions), "dates": list(dates), "n_days": Dd, "n_blocks": nb}


class SplitsAreLeakFree(unittest.TestCase):
    def test_daily_split_is_chronological_deduped_and_disjoint(self):
        from rl_quant.datasets import day_sequence, split_days
        b = [_fake_window(["2022-01-03", "2022-01-04", "2022-01-05"]),
             _fake_window(["2022-01-05", "2022-01-06"])]                 # 01-05 duplicated across windows
        self.assertEqual([d["date"] for d in day_sequence(b)],
                         ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"])  # sorted + deduped (keep-first)
        tr, va, te = split_days(b, "daily", 0.5, 0.25)
        sets = [set(d["date"] for d in s) for s in (tr, va, te)]
        self.assertEqual(sets[0] & sets[1], set())                       # NO date shared across splits (no leak)
        self.assertEqual(sets[0] & sets[2], set())
        self.assertEqual(sets[1] & sets[2], set())
        self.assertEqual(len(tr) + len(va) + len(te), 4)                 # every unique day used exactly once

    def test_intraday_split_is_window_level_and_disjoint(self):
        from rl_quant.datasets import split_days
        b = [_fake_window([f"2022-02-{i:02d}"]) for i in range(1, 9)]    # 8 disjoint 1-day windows
        tr, va, te = split_days(b, "intraday", 0.75, 0.10)
        sets = [set(d["date"] for d in s) for s in (tr, va, te)]
        self.assertEqual(sets[0] & sets[2], set())                       # train/test disjoint
        self.assertEqual(sets[0] & sets[1], set())
        self.assertEqual(len(tr) + len(va) + len(te), 8)


class StatisticalBatteryIsCorrect(unittest.TestCase):
    """Pin the credibility layer (the verdict reads it) to closed-form / known behavior so it can't silently regress."""

    def test_psr_half_at_benchmark_and_monotone(self):
        from rl_quant.evaluation.statistical import probabilistic_sharpe_ratio as psr
        self.assertAlmostEqual(psr(0.5, benchmark_sharpe=0.5, n_observations=100), 0.5, places=6)
        self.assertGreater(psr(0.6, benchmark_sharpe=0.0, n_observations=100),
                           psr(0.3, benchmark_sharpe=0.0, n_observations=100))   # monotone in observed Sharpe
        self.assertGreater(psr(0.3, benchmark_sharpe=0.0, n_observations=400),
                           psr(0.3, benchmark_sharpe=0.0, n_observations=100))   # monotone in n_observations

    def test_dsr_single_trial_is_psr_vs_zero_and_decreases_with_trials(self):
        from rl_quant.evaluation.statistical import deflated_sharpe_ratio as dsr
        from rl_quant.evaluation.statistical import probabilistic_sharpe_ratio as psr
        self.assertAlmostEqual(dsr(0.4, n_trials=1, n_observations=200),
                               psr(0.4, benchmark_sharpe=0.0, n_observations=200), places=6)
        self.assertGreater(dsr(0.4, n_trials=1, n_observations=200),
                           dsr(0.4, n_trials=50, n_observations=200))            # more trials -> lower credibility

    def test_effective_sample_size_drops_under_autocorrelation(self):
        from rl_quant.evaluation.statistical import effective_sample_size as ess
        g = torch.Generator().manual_seed(0)
        iid = torch.randn(2000, generator=g).tolist()
        rep = [v for v in torch.randn(200, generator=g).tolist() for _ in range(10)]  # block-repeated -> autocorr
        self.assertGreater(ess(iid), ess(rep))
        self.assertLess(ess(rep), len(rep))

    def test_wrc_spa_reject_dominant_positive_accept_negative(self):
        from rl_quant.evaluation.statistical import hansens_spa as spa
        from rl_quant.evaluation.statistical import white_reality_check as wrc
        g = torch.Generator().manual_seed(0)
        pos = [[0.01 + 0.001 * float(torch.randn(1, generator=g))] for _ in range(200)]   # one clearly +EV model
        g2 = torch.Generator().manual_seed(0)
        neg = [[-0.01 + 0.001 * float(torch.randn(1, generator=g2))] for _ in range(200)]  # clearly -EV
        self.assertLess(wrc(pos, n_bootstrap=400, seed=0), 0.2)          # low p -> edge survives snooping correction
        self.assertGreater(wrc(neg, n_bootstrap=400, seed=0), 0.4)       # no real edge -> high p
        self.assertLess(spa(pos, n_bootstrap=400, seed=0), 0.2)

    def test_block_bootstrap_ci_brackets_the_sample_mean(self):
        from rl_quant.evaluation.statistical import block_bootstrap_confidence_interval as ci
        g = torch.Generator().manual_seed(0)
        xs = (0.001 + 0.01 * torch.randn(500, generator=g)).tolist()
        lo, hi = ci(xs, statistic="mean", confidence=0.95, n_bootstrap=400, seed=0)
        m = sum(xs) / len(xs)
        self.assertLess(lo, m)
        self.assertGreater(hi, m)


if __name__ == "__main__":
    unittest.main()
