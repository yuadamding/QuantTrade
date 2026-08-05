"""Fast closed-loop mechanism checks for the Hold-30 chronology.

These tests exercise the package-owned delayed-fill runtime and local
origin-replay optimizer with tiny deterministic policies.  They are mechanism
qualification fixtures, not substitutes for the registered five-seed,
six-fold pre-lockbox experiment.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.datasets.hold30 import (
    Hold30NullDomain,
    n_time_transform,
    n_xs_transform,
)
from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.models.daily_policy import (
    HOLD30_HAZARD_MAX,
    HOLD30_HAZARD_MIN,
    Hold30Intent,
)
from rl_quant.training.hold30 import Hold30LossContract, train_hold30_update
from rl_quant.training.hold30_runtime import (
    Hold30ChronologicalReplayAdapter,
    Hold30ChronologicalRuntime,
    Hold30Sequence,
)

DTYPE = torch.float64
POSITIONS = 95
ORIGIN = 63
SOURCE_AXIS_ID = "1" * 64
RANDOMIZATION_AXIS_ID = "2" * 64


def _sequence(
    initial_weights: torch.Tensor,
    *,
    positions: int = POSITIONS,
    decision_state: torch.Tensor | None = None,
    asset_returns: torch.Tensor | None = None,
    benchmark_weights: torch.Tensor | None = None,
    cost_rate: float = 0.0,
    initial_age: int = 60,
    track_initial_units: bool = False,
) -> Hold30Sequence:
    weights = initial_weights.to(dtype=DTYPE)
    batch, assets = weights.shape
    if decision_state is None:
        decision_state = torch.zeros((positions, batch, assets, 1), dtype=DTYPE)
    if asset_returns is None:
        asset_returns = torch.zeros((positions - 1, batch, assets), dtype=DTYPE)
    masks = torch.ones((positions, batch, assets), dtype=torch.bool)
    if benchmark_weights is None:
        benchmark = weights.unsqueeze(0).expand(positions, -1, -1).clone()
    elif benchmark_weights.ndim == 2:
        benchmark = (
            benchmark_weights.to(DTYPE).unsqueeze(0).expand(positions, -1, -1).clone()
        )
    else:
        benchmark = benchmark_weights.to(DTYPE)
    benchmark_returns = (benchmark[:-1] * asset_returns).sum(dim=-1)
    return Hold30Sequence(
        decision_state=decision_state,
        asset_returns=asset_returns,
        decision_available=masks.clone(),
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=torch.ones_like(benchmark),
        risk_gross_max=torch.ones((positions, batch), dtype=DTYPE),
        benchmark_net_returns=benchmark_returns,
        initial_ledger=CohortLedger.from_weights(
            weights,
            cash_index=0,
            initial_age=initial_age,
            track_initial_units=track_initial_units,
        ),
        cost_rate=cost_rate,
        axis_id=SOURCE_AXIS_ID,
    )


class _CausalExposurePolicy(torch.nn.Module):
    """One scalar controls exposure only when asset-one's causal marker is on."""

    def __init__(self, exposure: float, *, entry_scale: float) -> None:
        super().__init__()
        self.exposure = torch.nn.Parameter(torch.tensor(exposure, dtype=DTYPE))
        self.entry_scale = float(entry_scale)

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del available, age_summaries
        signal = state_t[..., 0]
        return Hold30Intent(
            entry_scores=self.entry_scale * signal,
            hazard_residual=torch.full_like(prev_weights, HOLD30_HAZARD_MIN),
            exposure_residual=signal[:, 1] * self.exposure,
        )


class _NoisyExactHoldPolicy(torch.nn.Module):
    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del available, age_summaries
        return Hold30Intent(
            entry_scores=100.0 * state_t[..., 0],
            hazard_residual=torch.full_like(prev_weights, HOLD30_HAZARD_MIN),
            exposure_residual=torch.zeros(
                prev_weights.shape[0],
                dtype=prev_weights.dtype,
                device=prev_weights.device,
            ),
        )


class _StrongReversalPolicy(torch.nn.Module):
    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del state_t, available, age_summaries
        hazard = torch.full_like(prev_weights, HOLD30_HAZARD_MIN)
        hazard[:, 1:6] = HOLD30_HAZARD_MAX
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=hazard,
            exposure_residual=torch.zeros(
                prev_weights.shape[0],
                dtype=prev_weights.dtype,
                device=prev_weights.device,
            ),
        )


def _runtime() -> tuple[Hold30ChronologicalRuntime, Hold30ChronologicalReplayAdapter]:
    runtime = Hold30ChronologicalRuntime("H2")
    return runtime, Hold30ChronologicalReplayAdapter(runtime)


def _run(
    runtime: Hold30ChronologicalRuntime,
    policy: torch.nn.Module,
    sequence: Hold30Sequence,
):
    with torch.no_grad():
        return runtime.run_to_terminal(policy, sequence)


def _planted_sequence(
    *,
    positions: int = POSITIONS,
    initial_weights: torch.Tensor | None = None,
) -> Hold30Sequence:
    weights = (
        torch.tensor([[0.985, 0.001, 0.007, 0.007]], dtype=DTYPE)
        if initial_weights is None
        else initial_weights.to(dtype=DTYPE)
    )
    state = torch.zeros((positions, 1, 4, 1), dtype=DTYPE)
    state[ORIGIN, 0, 1, 0] = 1.0
    returns = torch.zeros((positions - 1, 1, 4), dtype=DTYPE)
    # The origin intent fills after row 63.  Rows 64..93 are exactly the 30
    # support returns that can reward the new cohort without lookahead.
    returns[ORIGIN + 1 : ORIGIN + 31, 0, 1] = 0.002
    return _sequence(
        weights,
        positions=positions,
        decision_state=state,
        asset_returns=returns,
    )


def _one_training_update(
    policy: _CausalExposurePolicy,
    sequence: Hold30Sequence,
    *,
    learning_rate: float,
) -> dict[str, object]:
    _runtime_instance, adapter = _runtime()
    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)
    return train_hold30_update(
        policy,
        sequence,
        adapter,
        optimizer,
        n_positions=sequence.n_positions,
        contract=Hold30LossContract.for_setting("hold30-m02-age-hazard"),
    )


def test_planted_signal_learns_larger_entry_and_survives_30_support_returns() -> None:
    sequence = _planted_sequence()
    policy = _CausalExposurePolicy(0.01, entry_scale=4.0)
    runtime, _adapter = _runtime()
    _before_terminal, before = _run(runtime, policy, sequence)
    before_entry = before[ORIGIN].filled_delta[0, 1].item()

    metrics = _one_training_update(policy, sequence, learning_rate=2.0)
    terminal, after = _run(runtime, policy, sequence)
    after_entry = after[ORIGIN].filled_delta[0, 1].item()
    support = after[ORIGIN + 1 : ORIGIN + 31]

    assert metrics["optimizer_steps"] == 1
    assert all(
        step.discretionary_accounting.turnover.item() == 0.0 for step in before[:ORIGIN]
    )
    assert after[ORIGIN].holding_return.item() == 0.0
    assert after_entry > 1.5 * before_entry > 0.0
    assert len(support) == 30
    assert all(step.discretionary_accounting.turnover.item() == 0.0 for step in support)
    assert all(
        step.turnover_by_cause[TurnoverCause.RISK_FORCED].item() == 0.0
        for step in support
    )
    # Only the newly purchased experimental cohort has retention units.  It
    # reaches age 30 intact after all thirty support returns.
    torch.testing.assert_close(
        terminal.ledger.retention_units[0, 1, 30],
        torch.tensor(after_entry, dtype=DTYPE),
    )
    assert terminal.ledger.economic_value[0, 1, 30].item() > after_entry


def test_one_session_score_noise_under_exact_hold_moves_at_most_half_percent() -> None:
    weights = torch.tensor([[0.985, 0.005, 0.005, 0.005]], dtype=DTYPE)
    state = torch.zeros((5, 1, 4, 1), dtype=DTYPE)
    state[1, 0, 1, 0] = 1.0
    sequence = _sequence(weights, positions=5, decision_state=state)
    runtime, _adapter = _runtime()

    terminal, transitions = _run(runtime, _NoisyExactHoldPolicy(), sequence)

    noisy_turnover = transitions[1].discretionary_accounting.turnover.item()
    assert noisy_turnover <= 0.005
    assert noisy_turnover == 0.0
    torch.testing.assert_close(terminal.ledger.weights, weights)


def test_strong_reversal_exits_at_least_80_percent_within_three_legal_fills() -> None:
    initial = torch.tensor([[0.95, *([0.01] * 5), *([0.0] * 5)]], dtype=DTYPE)
    replacement_benchmark = torch.tensor(
        [[0.95, *([0.0] * 5), *([0.01] * 5)]], dtype=DTYPE
    )
    sequence = _sequence(
        initial,
        positions=4,
        benchmark_weights=replacement_benchmark,
        initial_age=0,
        track_initial_units=True,
    )
    runtime, _adapter = _runtime()

    terminal, transitions = _run(runtime, _StrongReversalPolicy(), sequence)

    initial_mass = initial[0, 1:6].sum().item()
    remaining = terminal.ledger.weights[0, 1:6].sum().item()
    exit_fraction = 1.0 - remaining / initial_mass
    assert len(transitions) == 3
    assert exit_fraction >= 0.80
    assert transitions[0].discretionary_accounting.early_exit_notional.item() > 0.0
    assert all(
        step.turnover_by_cause[TurnoverCause.RISK_FORCED].item() == 0.0
        for step in transitions
    )


def test_tiny_null_cost_only_training_reduces_turnover_below_half_percent() -> None:
    weights = torch.tensor([[0.985, 0.005, 0.005, 0.005]], dtype=DTYPE)
    state = torch.zeros((POSITIONS, 1, 4, 1), dtype=DTYPE)
    state[ORIGIN, 0, 1, 0] = 1.0
    sequence = _sequence(weights, decision_state=state, cost_rate=0.20)
    policy = _CausalExposurePolicy(0.10, entry_scale=0.0)
    runtime, adapter = _runtime()
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)

    def origin_turnover() -> float:
        transition = _run(runtime, policy, sequence)[1][ORIGIN]
        return transition.discretionary_accounting.turnover.item()

    before = origin_turnover()
    assert before > 0.005
    updates = 0
    while updates < 4 and origin_turnover() > 0.005:
        train_hold30_update(
            policy,
            sequence,
            adapter,
            optimizer,
            n_positions=sequence.n_positions,
            contract=Hold30LossContract.for_setting("hold30-m02-age-hazard"),
        )
        updates += 1
    after = origin_turnover()

    assert 0 < updates <= 4
    assert after < before
    assert after <= 0.005


def test_registered_null_transforms_remove_tiny_planted_optimizer_edge() -> None:
    """Exercise N_time/N_xs with retraining, without claiming production evidence."""

    # Three 62-row roles are the smallest convenient synthetic partition with
    # a perfect one-to-one N_time matching at 31-position minimum separation.
    # Only asset one has benchmark support. The H2 benchmark-relative builder
    # can therefore buy the planted name, but cannot accidentally buy a
    # cross-sectionally permuted outcome through diffuse benchmark spillover.
    base = _planted_sequence(
        positions=187,
        initial_weights=torch.tensor([[0.995, 0.005, 0.0, 0.0]], dtype=DTYPE),
    )
    domains = (
        Hold30NullDomain("train", 0, 62),
        Hold30NullDomain("validation", 62, 124),
        Hold30NullDomain("outer", 124, 186),
    )
    ordinary = torch.ones_like(base.asset_returns, dtype=torch.bool)
    ordinary[..., 0] = False
    mandatory = torch.zeros_like(ordinary)
    active = torch.ones_like(ordinary)
    time_view = n_time_transform(
        base.asset_returns,
        ordinary,
        mandatory,
        active,
        domains=domains,
        seed=17,
        source_axis_id=SOURCE_AXIS_ID,
        randomization_axis_id=RANDOMIZATION_AXIS_ID,
    )
    cross_section_view = n_xs_transform(
        base.asset_returns,
        ordinary,
        mandatory,
        active,
        domains=domains,
        seed=29,
        source_axis_id=SOURCE_AXIS_ID,
        randomization_axis_id=RANDOMIZATION_AXIS_ID,
    )

    def rebuild_outcomes(asset_returns: torch.Tensor, axis_id: str) -> Hold30Sequence:
        # C1 is explicitly rebuilt from its held books and transformed return
        # path. Production must additionally rebuild labels, drift, and every
        # endpoint before the transformed dataset is admitted.
        weights = base.initial_ledger.weights
        sequence = _sequence(
            weights,
            positions=POSITIONS,
            decision_state=base.decision_state[:POSITIONS],
            asset_returns=asset_returns[: POSITIONS - 1],
        )
        return replace(sequence, axis_id=axis_id)

    planted_sequence = rebuild_outcomes(base.asset_returns, base.axis_id)
    time_sequence = rebuild_outcomes(
        time_view.asset_returns,
        time_view.receipt.transform_id,
    )
    cross_section_sequence = rebuild_outcomes(
        cross_section_view.asset_returns,
        cross_section_view.receipt.transform_id,
    )

    def trained_delta(sequence: Hold30Sequence) -> float:
        policy = _CausalExposurePolicy(0.01, entry_scale=4.0)
        initial = policy.exposure.item()
        _one_training_update(policy, sequence, learning_rate=2.0)
        return policy.exposure.item() - initial

    planted_delta = trained_delta(planted_sequence)
    time_delta = trained_delta(time_sequence)
    cross_section_delta = trained_delta(cross_section_sequence)

    assert time_view.receipt.kind == "N_time"
    assert cross_section_view.receipt.kind == "N_xs"
    torch.testing.assert_close(time_sequence.decision_state, planted_sequence.decision_state)
    torch.testing.assert_close(
        cross_section_sequence.decision_state,
        planted_sequence.decision_state,
    )
    assert not bool(time_sequence.asset_returns[ORIGIN + 1 : ORIGIN + 31, 0, 1].any())
    assert not bool(cross_section_sequence.asset_returns[:, 0, 1].any())
    assert planted_delta > 0.005
    assert time_delta == pytest.approx(0.0, abs=1e-14)
    assert cross_section_delta == pytest.approx(0.0, abs=1e-14)
