from __future__ import annotations

import torch

from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_ensemble import (
    aggregate_hold30_intents,
    decide_hold30_ensemble,
)


def test_h2_aggregates_outputs_not_member_portfolios() -> None:
    mask = torch.tensor([[True, True, True]])
    intents = tuple(
        Hold30Intent(
            entry_scores=torch.tensor(
                [[99.0, float(index - 2), -float(index - 2)]]
            ),
            hazard_residual=torch.full((1, 3), float(index - 2)),
            exposure_residual=torch.tensor([float(index)]),
        )
        for index in range(5)
    )
    result = aggregate_hold30_intents("H2", intents, mask)
    assert result.entry_scores is not None
    assert result.entry_scores[0, 0] == 0.0
    torch.testing.assert_close(result.entry_scores[0, 1:], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(result.hazard_residual, torch.zeros(1, 3))
    torch.testing.assert_close(result.exposure_residual, torch.tensor([2.0]))


def test_h0_uses_mean_centered_logits_and_median_gate() -> None:
    mask = torch.tensor([[True, True, False]])
    intents = tuple(
        Hold30Intent(
            target_logits=torch.tensor([[float(index), -float(index), 100.0]]),
            gate=torch.tensor([value]),
        )
        for index, value in enumerate((0.9, 0.1, 0.8, 0.2, 0.3))
    )
    result = aggregate_hold30_intents("H0", intents, mask)
    torch.testing.assert_close(result.gate, torch.tensor([0.3]))
    assert result.target_logits is not None
    # CASH is index zero and always participates; masked asset two is zeroed.
    torch.testing.assert_close(result.target_logits, torch.tensor([[2.0, -2.0, 0.0]]))


class _CountingPolicy(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.calls = 0

    def hold30_intent(self, state, weights, available, age):
        del state, available, age
        self.calls += 1
        return Hold30Intent(
            entry_scores=torch.full_like(weights, self.value),
            hazard_residual=torch.zeros_like(weights),
            exposure_residual=torch.full((weights.shape[0],), self.value),
        )


def test_ensemble_calls_each_member_once_on_the_shared_book() -> None:
    policies = tuple(_CountingPolicy(float(index)) for index in range(5))
    weights = torch.tensor([[0.8, 0.1, 0.1]])
    result = decide_hold30_ensemble(
        "H2",
        policies,
        torch.zeros(5, 1, 3, 4),
        weights,
        torch.ones_like(weights, dtype=torch.bool),
        torch.zeros(1, 3, 5),
    )
    assert len(result.member_intents) == 5
    assert [policy.calls for policy in policies] == [1, 1, 1, 1, 1]
    torch.testing.assert_close(result.aggregate_intent.exposure_residual, torch.tensor([2.0]))
