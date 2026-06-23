"""Synthetic economic-invariant suite for the minute-to-hour RL loop.

These are the deterministic "prove the loop is economically correct, not just non-crashing" tests requested by
the fundamental-correctness review: on tiny hand-built datasets where the right answer is known by construction,
the evaluation/environment accounting (and, for one case, the trained policy) must do the economically correct
thing. They complement the protocol invariants already enforced elsewhere (invalid returns are NaN, split
boundaries, return-basis agreement, weight-semantics fail-closed, missing-label gating) by pinning the END-TO-END
economic behaviour:

  * a profitable non-CASH action is actually captured and beats CASH,
  * a money-losing non-CASH action is correctly NOT worth holding (all-CASH is the correct outcome here),
  * a switch cost is charged against the payoff (a one-step edge below the switch cost is net-negative),
  * a requested non-CASH action with a missing realized label falls back to CASH AND is non-reportable,
  * a DQN trained on a trivially-profitable signal learns to prefer the profitable action over CASH.

Research/backtest only; no live trading.
"""

from __future__ import annotations

import dataclasses
import unittest

import torch

from rl_quant.core import DQNLearningConfig
from rl_quant.datasets.hour_from_second import default_second_to_hour_constraints
from rl_quant.envs.second_to_hour import (
    SecondToHourEnvConfig,
    transition_net_return_and_reward,
    transition_trade_cost_bps,
)
from rl_quant.second_to_hour_transformer import HourFromSecondDataSplit
from rl_quant.training.second_to_hour import (
    SecondToHourTrainingConfig,
    _ConstantActionModel,
    evaluate_second_to_hour_baselines,
    evaluate_second_to_hour_policy,
    train_second_to_hour_dqn,
)

CPU = torch.device("cpu")


def _zero_cost_constraints():
    """Default minute-to-hour constraints with all trading frictions zeroed (so a hold of a profitable action
    accrues its raw return with no leg/switch drag) -- lets a test isolate the return accounting from costs."""
    return dataclasses.replace(
        default_second_to_hour_constraints(),
        one_way_cost_bps=0.0,
        extra_switch_penalty_bps=0.0,
        cash_index=0,
    )


def _two_action_split(per_row_returns, *, name="eval", label_valid=None):
    """A minimal CASH/QQQ minute-to-hour split. ``per_row_returns`` is a list of ``[cash_return, qqq_return]``
    rows; every row is a contiguous, valid decision row (one episode). ``label_valid`` overrides the per-action
    label-valid mask (default all-valid). Invalid (NaN) returns are allowed -- that is the contract's
    representation of a missing label."""
    rows = len(per_row_returns)
    action_returns = torch.tensor(per_row_returns, dtype=torch.float32)
    label_valid_mask = (
        torch.ones((rows, 2), dtype=torch.bool)
        if label_valid is None
        else torch.as_tensor(label_valid, dtype=torch.bool)
    )
    return HourFromSecondDataSplit(
        name=name,
        decision_timestamps=[f"2026-01-02T{10 + i:02d}:30:00+00:00" for i in range(rows)],
        next_timestamps=[f"2026-01-02T{11 + i:02d}:30:00+00:00" for i in range(rows)],
        second_feature_names=["m"],
        hour_feature_names=["h"],
        action_names=["CASH", "QQQ"],
        second_features=torch.zeros((rows, 1, 1, 1)),
        second_mask=torch.ones((rows, 1, 1), dtype=torch.bool),
        hour_features=torch.zeros((rows, 1, 1)),
        action_returns=action_returns,
        action_valid_mask=torch.ones((rows, 2), dtype=torch.bool),
        label_valid_mask=label_valid_mask,
        valid_start_indices=torch.arange(rows, dtype=torch.long),
        valid_index_mask=torch.ones(rows, dtype=torch.bool),
        second_feature_mean=torch.zeros(1),
        second_feature_std=torch.ones(1),
        hour_feature_mean=torch.zeros(1),
        hour_feature_std=torch.ones(1),
        hours_lookback=1,
        seconds_per_hour=1,
    )


class SyntheticEconomicInvariants(unittest.TestCase):
    def test_profitable_non_cash_is_captured_and_beats_cash(self) -> None:
        # CASH return == 0, QQQ == +10bps every row, zero cost. Holding QQQ must (a) realize a clearly positive
        # return, (b) beat the always-CASH policy, and (c) actually be EXECUTED (a switch into QQQ), not silently
        # held in cash. If the loop were mis-wired (e.g. reward sign, action indexing), this fails.
        split = _two_action_split([[0.0, 0.001]] * 4)
        cons = _zero_cost_constraints()
        qqq = evaluate_second_to_hour_policy(
            split, _ConstantActionModel(2, 1), device=CPU, constraints=cons, cash_idle_penalty_bps=0.0
        )
        cash = evaluate_second_to_hour_policy(
            split, _ConstantActionModel(2, 0), device=CPU, constraints=cons, cash_idle_penalty_bps=0.0
        )
        self.assertAlmostEqual(cash.total_return, 0.0, places=9)
        self.assertGreater(qqq.total_return, 0.003)  # ~ (1.001^4 - 1) captured across the held rows
        self.assertGreater(qqq.total_return, cash.total_return)
        self.assertGreaterEqual(qqq.allocation_switches, 1)  # it actually entered QQQ
        self.assertTrue(qqq.evaluation_reportable)

    def test_money_losing_non_cash_makes_all_cash_the_correct_outcome(self) -> None:
        # QQQ == -10bps every row. Holding QQQ must lose money, so the always-CASH policy (0 return) is correctly
        # the better one. This proves an all-CASH result CAN be the economically correct answer -- the loop is
        # not simply biased toward (or against) trading.
        split = _two_action_split([[0.0, -0.001]] * 4)
        cons = _zero_cost_constraints()
        qqq = evaluate_second_to_hour_policy(
            split, _ConstantActionModel(2, 1), device=CPU, constraints=cons, cash_idle_penalty_bps=0.0
        )
        cash = evaluate_second_to_hour_policy(
            split, _ConstantActionModel(2, 0), device=CPU, constraints=cons, cash_idle_penalty_bps=0.0
        )
        self.assertLess(qqq.total_return, 0.0)
        self.assertAlmostEqual(cash.total_return, 0.0, places=9)
        self.assertGreater(cash.total_return, qqq.total_return)

    def test_switch_cost_is_charged_against_payoff(self) -> None:
        # The shared reward primitive must charge the transition cost against the raw return: a switch into an
        # action paying +100bps but costing 10bps nets +90bps of reward; a one-step +5bps edge does NOT cover a
        # 20bps switch (net-negative). If costs were dropped, a high-turnover policy would look free.
        cons = dataclasses.replace(_zero_cost_constraints(), one_way_cost_bps=10.0, extra_switch_penalty_bps=0.0)
        breakdown = transition_trade_cost_bps(
            torch.tensor([0]), torch.tensor([1]), constraints=cons, cash_idle_penalty_bps=0.0
        )
        net, reward = transition_net_return_and_reward(
            torch.tensor([0.01]), breakdown.trade_cost_bps, breakdown.cash_idle_bps, reward_scale=10_000.0
        )
        self.assertAlmostEqual(float(net[0]), 0.01 - 10.0 / 10_000.0, places=9)  # 100bps - 10bps = 90bps
        self.assertAlmostEqual(float(reward[0]), 90.0, places=4)
        # A +5bps edge under a 20bps switch is net-negative -> not worth trading for one step.
        small_net, _ = transition_net_return_and_reward(
            torch.tensor([0.0005]), torch.tensor([20.0]), torch.tensor([0.0]), reward_scale=10_000.0
        )
        self.assertLess(float(small_net[0]), 0.0)

    def test_missing_label_falls_back_to_cash_and_is_non_reportable(self) -> None:
        # A policy that REQUESTS QQQ on rows whose QQQ realized label is missing (NaN, label-invalid) must fall
        # back to CASH, count the requested-action missing label, and mark the evaluation NON-reportable -- so a
        # broken label pipeline cannot masquerade as a safe all-CASH "success".
        nan = float("nan")
        split = _two_action_split(
            [[0.0, nan], [0.0, nan], [0.0, nan]],
            label_valid=[[True, False], [True, False], [True, False]],
        )
        result = evaluate_second_to_hour_policy(
            split, _ConstantActionModel(2, 1), device=CPU, constraints=_zero_cost_constraints(),
            cash_idle_penalty_bps=0.0,
        )
        self.assertGreaterEqual(result.requested_action_missing_label_count, 1)
        self.assertFalse(result.evaluation_reportable)
        self.assertAlmostEqual(result.total_return, 0.0, places=9)  # executed CASH on every row

    def test_training_learns_to_prefer_profitable_action(self) -> None:
        # End-to-end: a DQN trained on a trivially-profitable signal (QQQ = +50bps every row, CASH = 0, zero
        # cost) must LEARN to prefer QQQ over CASH -- the trained greedy policy beats the always-CASH baseline.
        # Seeded for determinism; the signal is large and constant so a few hundred steps converge reliably.
        train = _two_action_split([[0.0, 0.005]] * 8, name="train")
        val = _two_action_split([[0.0, 0.005]] * 4, name="val")
        cons = _zero_cost_constraints()
        learning = DQNLearningConfig(
            num_envs=4, episode_length=4, replay_capacity=512, batch_size=32, train_steps=400, warmup_steps=32,
            gamma=0.9, learning_rate=3e-3, weight_decay=0.0, target_update_interval=25, epsilon_start=0.5,
            epsilon_end=0.0, eval_interval=1000, grad_clip=1.0,
        )
        config = SecondToHourTrainingConfig(
            env=SecondToHourEnvConfig(num_envs=4, episode_length=4, constraints=cons, cash_idle_penalty_bps=0.0),
            learning=learning, d_model=16, n_heads=2, second_layers=1, hour_layers=1, feedforward_dim=16,
            action_embedding_dim=4,
        )
        # Genuineness guard against a vacuous pass: a FRESH (untrained) model of the SAME architecture captures
        # nothing here -- it stays in CASH (empirically verified across seeds), so the trained policy's positive
        # return below is attributable to LEARNING, not initialization luck or a structural bias toward QQQ.
        from rl_quant.second_to_hour_transformer import SecondToHourPolicyQNetwork

        torch.manual_seed(0)
        untrained = SecondToHourPolicyQNetwork(
            second_feature_dim=1, hour_feature_dim=1, action_count=2, hours_lookback=1, seconds_per_hour=1,
            d_model=16, n_heads=2, second_layers=1, hour_layers=1, feedforward_dim=16, action_embedding_dim=4,
        )
        untrained_eval = evaluate_second_to_hour_policy(
            val, untrained, device=CPU, constraints=cons, cash_idle_penalty_bps=0.0
        )
        self.assertAlmostEqual(untrained_eval.total_return, 0.0, places=9)  # untrained captures nothing -> CASH

        torch.manual_seed(0)
        model, _artifacts = train_second_to_hour_dqn(train, val, device=CPU, config=config)
        trained = evaluate_second_to_hour_policy(
            val, model, device=CPU, constraints=cons, cash_idle_penalty_bps=0.0
        )
        baselines = evaluate_second_to_hour_baselines(
            val, device=CPU, constraints=cons, cash_idle_penalty_bps=0.0, include_buy_and_hold=False
        )
        always_cash = baselines["always_cash"]
        self.assertGreater(trained.total_return, always_cash.total_return)
        self.assertGreater(trained.total_return, untrained_eval.total_return)  # the improvement is LEARNED
        self.assertGreater(trained.total_return, 0.0)  # it captured the profitable edge
        self.assertGreaterEqual(trained.allocation_switches, 1)  # it actually traded into QQQ


if __name__ == "__main__":
    unittest.main()
