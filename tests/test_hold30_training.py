from __future__ import annotations

import math

import pytest
import torch

from rl_quant.training.hold30 import (
    Hold30CanonicalRow,
    Hold30LossContract,
    Hold30OriginReplay,
    Hold30ReplayGeometry,
    benchmark_relative_log_utility,
    calendar_diagnostic,
    detach_tree,
    origin_surrogate,
    sequence_coefficients,
    train_hold30_update,
)


def test_hold30_geometry_assigns_exactly_thirty_post_fill_returns() -> None:
    geometry = Hold30ReplayGeometry(max_origin_batch=2)
    roles = geometry.roles(100)
    roles.validate()

    assert roles.warmup.tolist() == list(range(63))
    assert roles.anchors.tolist() == list(range(63, 69))
    assert roles.support.tolist() == list(range(69, 99))
    assert roles.terminal_observation == 99
    assert roles.utility_rows[0].tolist() == list(range(63, 94))
    assert roles.replay_terminal_rows[0].item() == 94
    assert roles.utility_rows[-1].tolist() == list(range(68, 99))
    assert roles.replay_terminal_rows[-1].item() == 99
    assert roles.utility_mask.sum(1).tolist() == [31] * 6
    assert [batch.tolist() for batch in geometry.origin_batches(roles.anchors)] == [
        [63, 64],
        [65, 66],
        [67, 68],
    ]


def test_hold30_geometry_rejects_a_block_without_an_anchor() -> None:
    with pytest.raises(ValueError, match="at least 95 positions"):
        Hold30ReplayGeometry().roles(94)


def test_benchmark_relative_log_utility_is_additive_and_guarded() -> None:
    policy = torch.tensor([0.02, -0.01])
    benchmark = torch.tensor([0.01, -0.02])
    utility = benchmark_relative_log_utility(policy, benchmark)
    expected = torch.log1p(policy) - torch.log1p(benchmark)
    assert torch.equal(utility, expected)
    with pytest.raises(ValueError, match="greater than -1"):
        benchmark_relative_log_utility(torch.tensor([-1.0]), torch.tensor([0.0]))


def test_sequence_coefficients_and_h2_origin_terms_are_frozen() -> None:
    rows = [Hold30CanonicalRow(0.0, discretionary_turnover=0.05, gate=0.1) for _ in range(100)]
    anchors = Hold30ReplayGeometry().roles(100).anchors
    contract = Hold30LossContract.for_setting("hold30-m02-age-hazard")
    coefficients = sequence_coefficients(rows, anchors, contract)
    assert coefficients.anchor_count == 6
    assert coefficients.mean_turnover == pytest.approx(0.05)
    assert coefficients.turnover_coefficient == pytest.approx(2.0 * (0.05 - 1.0 / 30.0))
    assert coefficients.gate_coefficient == 0.0

    replay = Hold30OriginReplay(
        origin=63,
        utility_rows=torch.ones(31),
        discretionary_turnover=torch.tensor(0.04),
        early_sale_mass=torch.tensor(0.25),
        gate=torch.tensor(0.0),
        gate_entropy=torch.tensor(0.0),
    )
    value = origin_surrogate(replay, coefficients, contract)
    assert float(value) == pytest.approx(
        31.0 - coefficients.turnover_coefficient * 0.04 - 0.002 * 0.25
    )


def test_calendar_diagnostic_counts_each_anchor_once() -> None:
    rows = [
        Hold30CanonicalRow(
            utility=float(index) / 10_000.0,
            discretionary_turnover=0.04,
            early_sale_mass=0.25,
        )
        for index in range(100)
    ]
    anchors = Hold30ReplayGeometry().roles(100).anchors
    contract = Hold30LossContract.for_setting("hold30-m02-age-hazard")
    diagnostic = calendar_diagnostic(rows, anchors, contract)
    expected_mean = sum(float(index) / 10_000.0 for index in range(63, 69)) / 6.0
    expected_turnover_penalty = (0.04 - 1.0 / 30.0) ** 2

    assert diagnostic.anchor_count == 6
    assert diagnostic.mean_utility == pytest.approx(expected_mean)
    assert diagnostic.mean_discretionary_turnover == pytest.approx(0.04)
    assert diagnostic.turnover_penalty == pytest.approx(expected_turnover_penalty)
    assert diagnostic.mean_early_sale_mass == pytest.approx(0.25)
    assert diagnostic.early_exit_penalty == pytest.approx(0.002 * 0.25)
    assert diagnostic.value == pytest.approx(
        expected_mean - expected_turnover_penalty - 0.002 * 0.25
    )


def test_h0_uses_gate_budget_and_entropy_not_turnover() -> None:
    rows = [Hold30CanonicalRow(0.0, discretionary_turnover=0.9, gate=0.1) for _ in range(100)]
    anchors = Hold30ReplayGeometry().roles(100).anchors
    contract = Hold30LossContract.for_setting("hold30-m00-legacy-gate")
    coefficients = sequence_coefficients(rows, anchors, contract)
    assert coefficients.turnover_coefficient == 0.0
    assert coefficients.gate_coefficient == pytest.approx(1e-3)
    replay = Hold30OriginReplay(
        origin=63,
        utility_rows=torch.zeros(31),
        discretionary_turnover=torch.tensor(1.0),
        early_sale_mass=torch.tensor(1.0),
        gate=torch.tensor(0.2),
        gate_entropy=torch.tensor(0.5),
    )
    assert float(origin_surrogate(replay, coefficients, contract)) == pytest.approx(
        -1e-3 * 0.2 + 1e-5 * 0.5
    )


def test_detach_tree_preserves_values_and_breaks_graphs() -> None:
    source = torch.tensor([1.0], requires_grad=True)
    detached = detach_tree({"state": [source, (source + 1.0,)]})
    assert detached["state"][0].item() == 1.0
    assert detached["state"][1][0].item() == 2.0
    assert not detached["state"][0].requires_grad
    assert detached["state"][0].data_ptr() != source.data_ptr()


class _ScalarPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor(0.0))


class _DeterministicAdapter:
    def __init__(self) -> None:
        self.canonical_calls = 0
        self.replay_batches: list[list[int]] = []

    def canonical_pass(self, policy, sequence, roles):
        self.canonical_calls += 1
        assert not torch.is_grad_enabled()
        rows = [Hold30CanonicalRow(0.0, discretionary_turnover=0.04) for _ in range(99)]
        return {"book": torch.tensor([0.25], requires_grad=False)}, rows

    def replay_origins(self, policy, sequence, canonical_state, origins, roles):
        self.replay_batches.append(origins.tolist())
        assert not canonical_state["book"].requires_grad
        result = []
        for origin in origins.tolist():
            # The origin's total credited utility is exactly policy.score.
            utility = policy.score.expand(31) / 31.0
            result.append(
                Hold30OriginReplay(
                    origin=origin,
                    utility_rows=utility,
                    discretionary_turnover=policy.score * 0.0 + 0.04,
                    early_sale_mass=policy.score * 0.0,
                    gate=policy.score * 0.0,
                    gate_entropy=policy.score * 0.0,
                )
            )
        return result


def test_train_hold30_update_replays_each_anchor_once_and_steps_once() -> None:
    policy = _ScalarPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    adapter = _DeterministicAdapter()
    geometry = Hold30ReplayGeometry(max_origin_batch=2)
    metrics = train_hold30_update(
        policy,
        sequence=object(),
        adapter=adapter,
        optimizer=optimizer,
        n_positions=100,
        contract=Hold30LossContract.for_setting("hold30-a06-no-turn-penalty"),
        geometry=geometry,
    )

    assert adapter.canonical_calls == 1
    assert adapter.replay_batches == [[63, 64], [65, 66], [67, 68]]
    assert policy.score.item() == pytest.approx(0.1)
    assert metrics["anchor_count"] == 6
    assert metrics["origin_batch_count"] == 3
    assert metrics["utility_rows_replayed"] == 6 * 31
    assert metrics["optimizer_steps"] == 1
    assert metrics["calendar_objective"] == 0.0
    assert math.isfinite(metrics["objective"])
