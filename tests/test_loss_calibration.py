"""Loss-scale calibration guards (the 2026-06-29 audit).

The audit showed the old objective-shaping coefficients (budget 0.1, gate_ent 1e-3, missing 1.0) sat 2-4 orders of
magnitude above the per-step return scale: optimizing the loss with ZERO edge reproduced the trained run exactly
(gate pinned at max_actions/nB, cash 0.9+), and the marginal trade beyond budget had to clear ~budget_lambda ~= 9.5%
NET per 5-minute trade -- unreachable by any signal (IC<=1 tops out ~0.6%). These tests lock the recalibration:
penalties must be subordinate to a realistic edge, so a policy CAN express signal when it exists.
"""

from __future__ import annotations

import unittest

import torch

from rl_quant.training.decision_policy import _loss
from rl_quant.training.designs import Phase1Design


def _optimize_gate(edge_net: float, budget_lambda: float, gate_entropy_coef: float,
                   n_blocks: int = 78, max_actions: float = 5.0, steps: int = 800) -> float:
    """Minimize the REAL intraday `_loss` over free per-block gate logits when each unit of gate earns a fixed
    `edge_net` net return (the marginal-trade economics isolated from allocation learning). Returns mean gate."""
    logits = torch.nn.Parameter(torch.full((1, n_blocks), 2.0))     # gate_init_bias
    opt = torch.optim.Adam([logits], lr=5e-2)
    label = torch.ones(1, n_blocks, dtype=torch.bool)
    zeros = torch.zeros(1, n_blocks)
    for _ in range(steps):
        gates = torch.sigmoid(logits)
        nets = gates * edge_net                                     # net return accrues in proportion to the gate
        loss = _loss(nets, gates, zeros, zeros, label, risk_lambda=0.1, entropy_coef=0.0,
                     max_actions=max_actions, budget_lambda=budget_lambda,
                     gate_entropy_coef=gate_entropy_coef, missing_label_penalty=1e-3)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(torch.sigmoid(logits).mean())


class GateBudgetGeometry(unittest.TestCase):
    TARGET = 5.0 / 78.0                                             # max_actions/nB = the budget kink

    def test_recalibrated_budget_lets_a_real_edge_open_the_gate(self) -> None:
        """With return-unit coefficients, a 20bp/block net edge (~4x cost) must pull the gate WELL past the soft
        budget rate -- the budget is a preference, not a clamp."""
        g = _optimize_gate(edge_net=2e-3, budget_lambda=1e-3, gate_entropy_coef=1e-5)
        self.assertGreater(g, 2 * self.TARGET)

    def test_zero_edge_still_respects_the_budget(self) -> None:
        """With zero edge the recalibrated gate must stay at/below ~the budget rate (no free trading)."""
        g = _optimize_gate(edge_net=0.0, budget_lambda=1e-3, gate_entropy_coef=1e-5)
        self.assertLess(g, 2 * self.TARGET)

    def test_old_coefficients_pinned_the_gate_regardless_of_edge(self) -> None:
        """Regression documentation: under the OLD weights (budget 0.1, gate_ent 1e-3) even a 20bp/block edge
        could not move the gate past the kink -- the pathology the audit found in the trained run."""
        g = _optimize_gate(edge_net=2e-3, budget_lambda=0.1, gate_entropy_coef=1e-3)
        self.assertLess(g, 1.5 * self.TARGET)


class DefaultConsistency(unittest.TestCase):
    def test_design_defaults_are_return_unit_calibrated(self) -> None:
        d = Phase1Design("t", "t", session_seconds=23400, block_seconds=300, d_model=24, enc_layers=1,
                         enc_heads=2, policy_token_dim=24, policy_layers=1, policy_heads=2,
                         ssl_steps=1, policy_steps=1, ssl_batch_size=1, batch_days=1)
        # penalties live on the per-step return scale (~1e-4..1e-3), NOT orders of magnitude above it
        self.assertLessEqual(d.budget_lambda, 2e-3)
        self.assertLessEqual(d.gate_entropy_coef, 1e-4)
        self.assertLessEqual(d.missing_label_penalty, 1e-2)
        self.assertGreater(d.gate_entropy_coef, 0.0)                # keep >0: the path out of the CASH basin

    def test_trainer_defaults_match_design_defaults(self) -> None:
        """The trainer function defaults and Phase1Design defaults must not drift apart."""
        import inspect

        from rl_quant.training.daily_policy import train_daily_policy
        from rl_quant.training.decision_policy import train_decision_policy
        d = Phase1Design("t", "t", session_seconds=23400, block_seconds=300, d_model=24, enc_layers=1,
                         enc_heads=2, policy_token_dim=24, policy_layers=1, policy_heads=2,
                         ssl_steps=1, policy_steps=1, ssl_batch_size=1, batch_days=1)
        for fn in (train_decision_policy, train_daily_policy):
            sig = inspect.signature(fn)
            self.assertEqual(sig.parameters["budget_lambda"].default, d.budget_lambda, fn.__name__)
            self.assertEqual(sig.parameters["gate_entropy_coef"].default, d.gate_entropy_coef, fn.__name__)
            self.assertEqual(sig.parameters["missing_label_penalty"].default, d.missing_label_penalty, fn.__name__)


if __name__ == "__main__":
    unittest.main()
