from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.envs.hold30 import (
    CohortLedger,
    TurnoverCause,
    reconcile_cash_simplex_roundoff,
)
from rl_quant.execution.hold30 import Hold30BuiltAction, build_alpha_hold30_action
from rl_quant.models.daily_policy import (
    HOLD30_HAZARD_MIN,
    Hold30Intent,
    exact_hold30_intent,
)
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_EXIT_ACTION_COUNT,
    M03R_V6_EXIT_ACTION_INDEX,
    M03R_V6_HOLD_ACTION_INDEX,
    M03RV6ExitAction,
    straight_through_m03r_v6_exit_action,
)
from rl_quant.training import hold30_runtime
from rl_quant.training.hold30 import (
    Hold30LossContract,
    Hold30ReplayGeometry,
    train_hold30_update,
)
from rl_quant.training.hold30_runtime import (
    FunctionalHold30DecisionStateProvider,
    Hold30ChronologicalReplayAdapter,
    Hold30ChronologicalRuntime,
    Hold30SafetyProjectionError,
    Hold30Sequence,
)


def test_top2000_fp32_simplex_roundoff_is_repaired_before_accounting() -> None:
    """Regression for the failed 2xH100 TOP2000 training boundary."""

    risky_assets = 334
    weights = torch.zeros((1, risky_assets + 1), dtype=torch.float32)
    weights[:, 1:] = 1.0 / risky_assets
    # This exact reduction order generated the production failure.
    assert float(1.0 - weights[:, 1:].sum()) == pytest.approx(
        -1.1920928955078125e-07,
        abs=0.0,
    )

    projected = hold30_runtime._risk_project(
        weights,
        torch.full_like(weights, 0.01),
        torch.ones(1, dtype=torch.float32),
        0,
    )
    assert bool((projected >= 0).all())
    torch.testing.assert_close(
        projected.sum(-1),
        torch.ones(1, dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert float(projected[:, 1:].sum()) <= 1.0

    ledger = CohortLedger.from_weights(projected, cash_index=0)
    unchanged, accounting = ledger.trade_to(
        ledger.weights,
        cause=TurnoverCause.RISK_FORCED,
    )
    assert float(accounting.turnover.item()) == 0.0
    assert bool((unchanged.weights >= 0).all())
    torch.testing.assert_close(
        unchanged.weights.sum(-1),
        torch.ones(1, dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )

    small_sale_target = ledger.weights.clone()
    small_sale_target[:, 1] -= 1.0e-4
    small_sale_target[:, 0] += 1.0e-4
    after_sale, sale_accounting = ledger.trade_to(
        small_sale_target,
        cause=TurnoverCause.DISCRETIONARY,
    )
    after_sale.assert_reconciles(small_sale_target)
    assert sale_accounting.turnover.item() == pytest.approx(1.0e-4)


def test_top2000_current_shape_survives_drift_and_zero_change_trade() -> None:
    risky_assets = 1_998
    weights = torch.zeros((1, risky_assets + 1), dtype=torch.float32)
    weights[:, 1:] = 1.0 / risky_assets
    ledger = CohortLedger.from_weights(weights, cash_index=0)
    generator = torch.Generator().manual_seed(3)
    returns = torch.zeros_like(weights)
    returns[:, 1:] = 0.001 * torch.randn(
        (1, risky_assets),
        generator=generator,
    )

    drifted = ledger.age_and_drift(returns)
    assert bool((drifted.weights >= 0).all())
    unchanged, accounting = drifted.trade_to(
        drifted.weights,
        cause=TurnoverCause.DISCRETIONARY,
    )
    assert float(accounting.turnover.item()) == 0.0
    assert bool((unchanged.weights >= 0).all())


@pytest.mark.parametrize("risky_assets", [1_602, 1_626, 1_654, 1_700, 1_750])
def test_staggered_top2000_ledger_reconciles_after_normal_sale(
    risky_assets: int,
) -> None:
    weights = torch.zeros((1, 1_999), dtype=torch.float32)
    weights[:, 1 : risky_assets + 1] = 1.0 / risky_assets
    ledger = CohortLedger.from_staggered_endowment(
        weights,
        cash_index=0,
        youngest_age=0,
        oldest_age=29,
    )
    target = ledger.weights.clone()
    target[:, 1] -= 1.0e-4
    target[:, 0] += 1.0e-4

    after_sale, accounting = ledger.trade_to(
        target,
        cause=TurnoverCause.DISCRETIONARY,
    )
    after_sale.assert_reconciles(target)
    assert accounting.turnover.item() == pytest.approx(1.0e-4)


def test_simplex_reconciliation_rejects_material_overshoot_and_low_precision() -> None:
    with pytest.raises(ValueError, match="simplex|beyond numerical roundoff"):
        hold30_runtime._force_mask_to_cash(
            torch.tensor([[0.0, 0.55, 0.46]], dtype=torch.float32),
            torch.ones((1, 3), dtype=torch.bool),
            0,
        )
    with pytest.raises(TypeError, match="float32 or float64"):
        CohortLedger.from_weights(
            torch.tensor([[0.5, 0.5]], dtype=torch.bfloat16),
            cash_index=0,
        )
    for malformed in (
        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[-0.5, 1.0]], dtype=torch.float32),
        torch.tensor([[0.5, 0.1]], dtype=torch.float32),
    ):
        with pytest.raises(ValueError):
            reconcile_cash_simplex_roundoff(malformed, cash_index=0)


@pytest.mark.parametrize(
    "risky_assets",
    [100, 121, 300, 334, 1_000, 1_985, 1_994, 1_998],
)
def test_simplex_reconciliation_covers_top2000_asset_counts(
    risky_assets: int,
) -> None:
    weights = torch.zeros((1, risky_assets + 1), dtype=torch.float32)
    weights[:, 1:] = 1.0 / risky_assets
    reconciled = reconcile_cash_simplex_roundoff(weights, cash_index=0)

    assert bool((reconciled >= 0).all())
    torch.testing.assert_close(
        reconciled.sum(-1),
        torch.ones(1, dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def _sequence(
    positions: int,
    *,
    initial_weights: torch.Tensor | None = None,
    initial_age: int = 0,
    cost_rate: float = 0.0,
) -> Hold30Sequence:
    dtype = torch.float64
    weights = (
        torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=dtype)
        if initial_weights is None
        else initial_weights.to(dtype=dtype)
    )
    batch, assets = weights.shape
    state = torch.zeros((positions, batch, assets, 1), dtype=dtype)
    state[:, :, :, 0] = torch.arange(positions, dtype=dtype).view(-1, 1, 1)
    returns = torch.zeros((positions - 1, batch, assets), dtype=dtype)
    masks = torch.ones((positions, batch, assets), dtype=torch.bool)
    benchmark = weights.unsqueeze(0).expand(positions, -1, -1).clone()
    caps = torch.ones_like(benchmark)
    gross = torch.ones((positions, batch), dtype=dtype)
    benchmark_returns = torch.zeros((positions - 1, batch), dtype=dtype)
    return Hold30Sequence(
        decision_state=state,
        asset_returns=returns,
        decision_available=masks.clone(),
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        benchmark_net_returns=benchmark_returns,
        initial_ledger=CohortLedger.from_weights(
            weights,
            cash_index=0,
            initial_age=initial_age,
            track_initial_units=True,
        ),
        cost_rate=cost_rate,
        axis_id="fixture-axis-v1",
    )


class _ExactHoldPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        self.calls.append(
            (
                state_t.detach().clone(),
                prev_weights.detach().clone(),
                available.detach().clone(),
            )
        )
        return exact_hold30_intent(prev_weights)


class _V6ExactActionPolicy(torch.nn.Module):
    def __init__(self, action_index: int, hazard_residual: float) -> None:
        super().__init__()
        self.action_index = action_index
        self.hazard_residual = hazard_residual
        self.action_logits = torch.nn.Parameter(
            torch.tensor([0.0, -1.0, -1.0], dtype=torch.float64)
        )

    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        del state_t, age_summaries
        batch, assets = prev_weights.shape
        logits = self.action_logits.view(1, 1, M03R_V6_EXIT_ACTION_COUNT).expand(
            batch, assets, -1
        )
        selected = logits.clone()
        selected[:, 1, self.action_index] = 2.0
        risky = available.bool().clone()
        risky[:, 0] = False
        soft, decision = straight_through_m03r_v6_exit_action(selected)
        hold = torch.zeros_like(soft)
        hold[..., M03R_V6_HOLD_ACTION_INDEX] = 1.0
        soft = torch.where(risky.unsqueeze(-1), soft, hold)
        decision = torch.where(risky.unsqueeze(-1), decision, hold)
        action = M03RV6ExitAction(
            logits=selected,
            soft_probabilities=soft,
            decision_st=decision,
            risky_available=risky,
            exact_hold_atom_enabled=True,
        )
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=torch.full_like(prev_weights, self.hazard_residual),
            exposure_residual=torch.zeros(
                batch,
                dtype=prev_weights.dtype,
                device=prev_weights.device,
            ),
            exit_action_v6=action,
        )


@pytest.mark.parametrize(
    ("action_index", "hazard_residual", "expected_weight"),
    (
        (M03R_V6_HOLD_ACTION_INDEX, 12.0, 0.01),
        (M03R_V6_EXIT_ACTION_INDEX, -12.0, 0.0),
    ),
)
def test_v6_exact_action_survives_delayed_state_and_executes_at_fill(
    action_index: int,
    hazard_residual: float,
    expected_weight: float,
) -> None:
    initial = torch.tensor([[0.99, 0.01, 0.0, 0.0]], dtype=torch.float64)
    sequence = _sequence(2, initial_weights=initial)
    runtime = Hold30ChronologicalRuntime("H2")
    policy = _V6ExactActionPolicy(action_index, hazard_residual)

    pending = runtime.decide(policy, sequence, runtime.initial_state(sequence))
    assert pending.pending_intent is not None
    live_action = pending.pending_intent.intent.exit_action_v6
    assert live_action is not None
    assert live_action.decision_st[0, 1, action_index].item() == 1.0

    restored = pending.detach()
    assert restored.pending_intent is not None
    restored_action = restored.pending_intent.intent.exit_action_v6
    assert restored_action is not None
    assert restored_action.logits.data_ptr() != live_action.logits.data_ptr()
    assert not restored_action.logits.requires_grad

    next_state, transition = runtime.advance(sequence, restored)
    assert next_state.ledger.weights[0, 1].item() == pytest.approx(expected_weight)
    expected_sale = 0.01 - expected_weight
    assert transition.discretionary_accounting.net_sells[0, 1].item() == pytest.approx(
        expected_sale
    )


def test_v6_exact_exit_gradient_survives_pending_intent_until_fill() -> None:
    initial = torch.tensor([[0.99, 0.01, 0.0, 0.0]], dtype=torch.float64)
    sequence = _sequence(2, initial_weights=initial)
    runtime = Hold30ChronologicalRuntime("H2")
    policy = _V6ExactActionPolicy(M03R_V6_EXIT_ACTION_INDEX, -12.0)

    pending = runtime.decide(policy, sequence, runtime.initial_state(sequence))
    _next_state, transition = runtime.advance(sequence, pending)
    transition.pre_cost_weights[0, 1].backward()

    gradient = policy.action_logits.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient != 0.0).any())


def test_v6_exact_hold_is_not_overridden_by_discretionary_risk_overlay() -> None:
    weights = torch.tensor([[0.99, 0.01, 0.0, 0.0]], dtype=torch.float64)
    age_notional = torch.zeros((1, 4, 61), dtype=torch.float64)
    age_notional[0, 0, 5] = 0.99
    age_notional[0, 1, 5] = 0.01
    available = torch.ones_like(weights, dtype=torch.bool)
    policy = _V6ExactActionPolicy(M03R_V6_HOLD_ACTION_INDEX, 12.0)
    intent = policy.hold30_intent(
        torch.zeros((1, 4, 1), dtype=torch.float64),
        weights,
        available,
    )
    assert intent.exit_action_v6 is not None

    built = build_alpha_hold30_action(
        weights,
        age_notional,
        torch.zeros_like(weights),
        torch.full_like(weights, 12.0),
        torch.tensor([0.04], dtype=torch.float64),
        weights,
        available,
        torch.ones_like(weights),
        torch.ones(1, dtype=torch.float64),
        exit_action_v6=intent.exit_action_v6,
        total_risk_overlay=torch.tensor([-10.0], dtype=torch.float64),
        total_risk_step=0.10,
    )

    assert built.proposed_release[0, 1].item() == 0.0
    assert built.target_weights[0, 1].item() == pytest.approx(0.01)


def test_actor_cannot_see_future_return_or_fill_repair_and_new_fill_does_not_earn_old_return() -> None:
    sequence = _sequence(3)
    sequence.asset_returns[0, 0, 1] = 1.0
    sequence.fill_membership[1, 0, 2] = False
    policy = _ExactHoldPolicy()
    runtime = Hold30ChronologicalRuntime("H2")

    state = runtime.initial_state(sequence)
    pending = runtime.decide(policy, sequence, state)

    # The decision saw the original one-percent position and decision-time
    # mask, not the doubled future value or the fill-time deletion.
    assert len(policy.calls) == 1
    torch.testing.assert_close(policy.calls[0][1], torch.tensor([[0.97, 0.01, 0.01, 0.01]], dtype=torch.float64))
    assert bool(policy.calls[0][2][0, 2])

    next_state, transition = runtime.advance(sequence, pending)
    assert transition.holding_return.item() == pytest.approx(0.01)
    assert transition.execution_pretrade_weights[0, 1].item() == pytest.approx(0.02 / 1.01)
    assert transition.membership_repaired_weights[0, 2].item() == 0.0
    # Exact-hold H2 leaves the already repaired future book unchanged.
    torch.testing.assert_close(next_state.ledger.weights, transition.risk_repaired_weights)


def test_fill_order_types_forced_turnover_and_charges_cost_once() -> None:
    sequence = _sequence(2, cost_rate=0.002)
    sequence.fill_membership[1, 0, 1] = False
    sequence.fill_availability[1, 0, 2] = False
    sequence.risk_asset_caps[1, 0, 3] = 0.005
    runtime = Hold30ChronologicalRuntime("H2")

    state = runtime.decide(_ExactHoldPolicy(), sequence, runtime.initial_state(sequence))
    next_state, transition = runtime.advance(sequence, state)

    assert transition.turnover_by_cause[TurnoverCause.MEMBERSHIP_FORCED].item() == pytest.approx(0.01)
    assert transition.turnover_by_cause[TurnoverCause.AVAILABILITY_FORCED].item() == pytest.approx(0.01)
    assert transition.turnover_by_cause[TurnoverCause.RISK_FORCED].item() == pytest.approx(0.005)
    assert transition.turnover_by_cause[TurnoverCause.DISCRETIONARY].item() == 0.0
    assert transition.turnover_by_cause[TurnoverCause.STARTUP].item() == 0.0
    assert transition.turnover_by_cause[TurnoverCause.TERMINAL].item() == 0.0
    assert transition.discretionary_accounting.early_exit_notional.item() == 0.0
    forced_sold_units = sum(
        transition.accounting_by_cause[cause].sold_units_by_age
        for cause in (
            TurnoverCause.MEMBERSHIP_FORCED,
            TurnoverCause.AVAILABILITY_FORCED,
            TurnoverCause.RISK_FORCED,
        )
    )
    torch.testing.assert_close(
        transition.retention_units_before_membership - forced_sold_units,
        transition.retention_units_after_forced,
    )
    assert transition.cost.item() == pytest.approx(0.002 * 0.025)
    assert transition.cost_financing.sum().item() == pytest.approx(transition.cost.item())
    assert transition.net_return.item() == pytest.approx(-0.002 * 0.025)
    # Cost normalization is a distinct NAV stage, never a trade or projection.
    torch.testing.assert_close(transition.pre_cost_weights, transition.post_cost_weights)
    assert transition.pre_cost_weights.data_ptr() != transition.post_cost_weights.data_ptr()
    assert transition.projection_distance.item() == 0.0
    assert next_state.equity.item() == pytest.approx(1.0 - 0.002 * 0.025)


def test_book_and_cohort_age_carry_without_reset_or_terminal_liquidation() -> None:
    sequence = _sequence(6, initial_age=0)
    runtime = Hold30ChronologicalRuntime("H2")

    terminal, transitions = runtime.run_to_terminal(_ExactHoldPolicy(), sequence)

    assert len(transitions) == 5
    assert terminal.position_index == 5
    assert terminal.pending_intent is None
    torch.testing.assert_close(terminal.ledger.weights, sequence.initial_ledger.weights)
    assert terminal.ledger.economic_value[0, 1, 5].item() == pytest.approx(0.01)
    assert terminal.ledger.retention_units[0, 1, 5].item() == pytest.approx(0.01)
    assert terminal.ledger.weights[0, 1].item() > 0.0


def test_pending_intent_is_part_of_exact_resume_state() -> None:
    sequence = _sequence(5, cost_rate=0.002)
    sequence.asset_returns[:, 0, 1] = torch.tensor([0.01, -0.02, 0.03, 0.005], dtype=torch.float64)
    runtime = Hold30ChronologicalRuntime("H2")
    policy = _ExactHoldPolicy()

    after_decision = runtime.decide(policy, sequence, runtime.initial_state(sequence))
    checkpoint = after_decision.detach()
    assert checkpoint.pending_intent is not None
    assert checkpoint.pending_intent.decision_index == 0
    assert checkpoint.pending_intent.fill_index == 1
    assert checkpoint.pending_intent.axis_id == sequence.axis_id

    expected, _ = runtime.run_to_terminal(policy, sequence, after_decision)
    resumed, _ = runtime.run_to_terminal(policy, sequence, checkpoint)
    torch.testing.assert_close(resumed.equity, expected.equity)
    torch.testing.assert_close(resumed.ledger.economic_value, expected.ledger.economic_value)
    torch.testing.assert_close(resumed.ledger.retention_units, expected.ledger.retention_units)


class _UpstreamStatePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.call_count = 0

    def encode_state(self, raw_state):
        return raw_state * self.encoder_scale

    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        self.call_count += 1
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=torch.full_like(prev_weights, HOLD30_HAZARD_MIN),
            exposure_residual=state_t[:, 0, 0],
        )


def test_canonical_replay_matches_and_credits_exactly_thirty_post_fill_returns() -> None:
    initial = torch.tensor([[0.985, 0.005, 0.005, 0.005]], dtype=torch.float64)
    sequence = _sequence(95, initial_weights=initial)
    sequence.decision_state.zero_()
    sequence.decision_state[63] = 1.0
    # Origin 63 fills after row-63's already-completed zero return.  Only rows
    # 64..93 contain the thirty returns earned by that newly filled book.
    sequence.asset_returns[64:94, 0, 1] = 0.001
    geometry = Hold30ReplayGeometry()
    roles = geometry.roles(sequence.n_positions)
    provider = FunctionalHold30DecisionStateProvider(
        canonical_fn=lambda policy, bound: policy.encode_state(bound.decision_state[:-1]),
        replay_origin_fn=lambda policy, bound, origin: policy.encode_state(
            bound.decision_state[origin]
        ),
    )
    runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=provider,
        require_trainable_state_provider=True,
    )
    adapter = Hold30ChronologicalReplayAdapter(runtime)
    policy = _UpstreamStatePolicy()

    with torch.no_grad():
        trace, rows = adapter.canonical_pass(policy, sequence, roles)
    canonical_calls = policy.call_count
    replay = adapter.replay_origins(policy, sequence, trace, roles.anchors, roles)[0]

    assert len(rows) == 94
    assert len(trace.boundary_states) == 95
    assert len(trace.pending_intents) == 94
    assert replay.origin == 63
    assert replay.utility_rows.numel() == 31
    assert replay.discretionary_sold_value_by_age is not None
    assert replay.discretionary_sold_value_by_age.shape == (61,)
    assert replay.discretionary_sold_value_by_age.sum() >= 0
    assert policy.call_count == canonical_calls + 1  # support intents were reused, not reevaluated
    first_grad = torch.autograd.grad(
        replay.utility_rows[0], policy.encoder_scale, retain_graph=True
    )[0]
    tail_grad = torch.autograd.grad(replay.utility_rows[1:].sum(), policy.encoder_scale)[0]
    assert first_grad.item() == pytest.approx(0.0, abs=1e-14)
    assert abs(tail_grad.item()) > 0.0
    assert trace.terminal_state.ledger.weights[0, 1].item() > initial[0, 1].item()


def test_runtime_consumes_one_batched_origin_state_call_and_fails_closed_on_shape() -> None:
    sequence = _sequence(96)
    roles = Hold30ReplayGeometry().roles(sequence.n_positions)
    calls = {"scalar": 0, "batch": 0}

    def scalar_state(policy, bound, origin):
        del policy, bound, origin
        calls["scalar"] += 1
        raise AssertionError("runtime must not request scalar origin states")

    def batched_states(policy, bound, origins):
        calls["batch"] += 1
        index = origins.to(device=bound.decision_state.device)
        return policy.encode_state(bound.decision_state.index_select(0, index))

    provider = FunctionalHold30DecisionStateProvider(
        canonical_fn=lambda policy, bound: policy.encode_state(bound.decision_state[:-1]),
        replay_origin_fn=scalar_state,
        replay_origin_states_fn=batched_states,
    )
    runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=provider,
        require_trainable_state_provider=True,
    )
    policy = _UpstreamStatePolicy()
    with torch.no_grad():
        trace, _rows = runtime.canonical_pass(policy, sequence, roles)

    replays = runtime.replay_origins(policy, sequence, trace, roles.anchors, roles)

    assert [replay.origin for replay in replays] == [63, 64]
    assert calls == {"scalar": 0, "batch": 1}

    malformed_provider = FunctionalHold30DecisionStateProvider(
        canonical_fn=lambda bound_policy, bound: bound_policy.encode_state(
            bound.decision_state[:-1]
        ),
        replay_origin_fn=scalar_state,
        replay_origin_states_fn=lambda bound_policy, bound, origins: bound_policy.encode_state(
            bound.decision_state.index_select(0, origins[:1])
        ),
    )
    malformed_runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=malformed_provider,
        require_trainable_state_provider=True,
    )
    with pytest.raises(ValueError, match="replay tensor must have shape"):
        malformed_runtime.replay_origins(
            policy,
            sequence,
            trace,
            roles.anchors,
            roles,
        )


def test_training_mode_rejects_a_precomputed_state_provider() -> None:
    with pytest.raises(ValueError, match="cannot train the upstream encoder"):
        Hold30ChronologicalRuntime("H2", require_trainable_state_provider=True)


def test_real_adapter_optimizer_step_updates_the_upstream_encoder_once() -> None:
    initial = torch.tensor([[0.985, 0.005, 0.005, 0.005]], dtype=torch.float64)
    sequence = _sequence(95, initial_weights=initial)
    sequence.decision_state.zero_()
    sequence.decision_state[63] = 1.0
    sequence.asset_returns[64:94, 0, 1] = 0.001
    provider = FunctionalHold30DecisionStateProvider(
        canonical_fn=lambda policy, bound: policy.encode_state(bound.decision_state[:-1]),
        replay_origin_fn=lambda policy, bound, origin: policy.encode_state(
            bound.decision_state[origin]
        ),
    )
    runtime = Hold30ChronologicalRuntime(
        "H2",
        state_provider=provider,
        require_trainable_state_provider=True,
    )
    policy = _UpstreamStatePolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.encoder_scale.detach().clone()

    metrics = train_hold30_update(
        policy,
        sequence,
        Hold30ChronologicalReplayAdapter(runtime),
        optimizer,
        n_positions=sequence.n_positions,
        contract=Hold30LossContract.for_setting("hold30-a06-no-turn-penalty"),
    )

    assert metrics["optimizer_steps"] == 1
    assert metrics["anchor_count"] == 1
    assert not torch.equal(policy.encoder_scale.detach(), before)


class _UnsafeBuilder:
    def __call__(self, intent, repaired_ledger, benchmark, trade_mask, caps, gross):
        del intent, benchmark, trade_mask, caps, gross
        weights = repaired_ledger.weights
        target = weights.clone()
        target[:, 1] = target[:, 1] + 0.01
        target[:, 0] = target[:, 0] - 0.01
        delta = target - weights
        turnover = 0.5 * delta.abs().sum(-1)
        return Hold30BuiltAction(
            target_weights=target,
            requested_delta=delta,
            constructed_delta=delta,
            requested_turnover=turnover,
            constructed_turnover=turnover,
            desired_risky_exposure=target[:, 1:].sum(-1),
            proposed_release_by_age=weights.new_zeros((*weights.shape, 61)),
            proposed_release=torch.zeros_like(weights),
            capacity_shortfall=torch.zeros_like(turnover),
        )


class _SleevePolicy(torch.nn.Module):
    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        del state_t, available, age_summaries
        return Hold30Intent(entry_scores=torch.zeros_like(prev_weights))


class _EntryTrackingPolicy(torch.nn.Module):
    def hold30_intent(self, state_t, prev_weights, available, age_summaries=None):
        del available, age_summaries
        decision = int(state_t[0, 0, 0])
        hazard = torch.full_like(prev_weights, HOLD30_HAZARD_MIN)
        exposure = prev_weights.new_full((prev_weights.shape[0],), 10.0 if decision == 0 else 0.0)
        if decision == 1:
            hazard[:, 1] = 12.0
        return Hold30Intent(
            entry_scores=torch.zeros_like(prev_weights),
            hazard_residual=hazard,
            exposure_residual=exposure,
        )


def test_support_purchases_have_economic_age_but_do_not_enter_score_origin_units() -> None:
    initial = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    sequence = _sequence(3, initial_weights=initial)
    sequence.benchmark_weights.zero_()
    sequence.benchmark_weights[0, :, 0] = 1.0
    sequence.benchmark_weights[1, :, 0] = 0.99
    sequence.benchmark_weights[1, :, 1] = 0.01
    sequence.benchmark_weights[2, :, 0] = 0.99
    sequence.benchmark_weights[2, :, 2] = 0.01
    sequence = replace(
        sequence,
        track_entry_units=torch.tensor([True, False], dtype=torch.bool),
    )
    runtime = Hold30ChronologicalRuntime("H2")

    terminal, transitions = runtime.run_to_terminal(_EntryTrackingPolicy(), sequence)

    score_fill, support_fill = transitions
    assert score_fill.discretionary_accounting.entry_units_added[0, 1].item() == pytest.approx(0.01)
    assert support_fill.retention_units_before_membership[0, 1, 1].item() == pytest.approx(0.01)
    assert support_fill.discretionary_accounting.sold_units_by_age[0, 1, 1].item() > 0.0
    assert terminal.ledger.economic_value[0, 2, 0].item() > 0.0
    assert terminal.ledger.retention_units[0, 2, 0].item() == 0.0
    assert 0.0 < terminal.ledger.retention_units[0, 1, 1].item() < 0.01


def test_h3_runtime_carries_exact_sleeve_phase_and_reconciles_shared_ledger() -> None:
    initial = torch.zeros((1, 101), dtype=torch.float64)
    initial[:, 0] = 1.0
    sequence = _sequence(32, initial_weights=initial)
    sequence.benchmark_weights.zero_()
    sequence.benchmark_weights[:, :, 1:] = 0.01
    runtime = Hold30ChronologicalRuntime("H3")

    terminal, transitions = runtime.run_to_terminal(_SleevePolicy(), sequence)

    assert transitions[0].sleeve_review is not None
    assert transitions[0].sleeve_review.maturing_sleeve == 0
    assert transitions[0].sleeve_review.review_age == 30
    assert transitions[30].sleeve_review is not None
    assert transitions[30].sleeve_review.maturing_sleeve == 0
    assert transitions[30].sleeve_review.review_age == 30
    assert terminal.sleeve_snapshot is not None
    assert terminal.sleeve_snapshot.session_index == 31
    torch.testing.assert_close(
        terminal.sleeve_snapshot.books.sum(dim=1),
        terminal.ledger.weights,
        atol=1e-12,
        rtol=0.0,
    )
    assert all(item.projection_distance.item() <= 1e-12 for item in transitions)


def test_h3_canonical_replay_restores_identical_sleeve_state() -> None:
    sequence = _sequence(95)
    runtime = Hold30ChronologicalRuntime("H3")
    adapter = Hold30ChronologicalReplayAdapter(runtime)
    roles = Hold30ReplayGeometry().roles(sequence.n_positions)

    with torch.no_grad():
        trace, _rows = adapter.canonical_pass(_SleevePolicy(), sequence, roles)
    replay = adapter.replay_origins(
        _SleevePolicy(), sequence, trace, roles.anchors, roles
    )[0]

    assert replay.utility_rows.numel() == 31
    assert trace.boundary_states[63].sleeve_snapshot is not None
    assert trace.boundary_states[94].sleeve_snapshot is not None
    assert trace.boundary_states[94].sleeve_snapshot.session_index == 94


def test_training_safety_projection_fails_closed_above_one_micro_weight() -> None:
    sequence = _sequence(2)
    runtime = Hold30ChronologicalRuntime("H2", action_builder=_UnsafeBuilder())
    state = runtime.decide(_ExactHoldPolicy(), sequence, runtime.initial_state(sequence))

    with pytest.raises(Hold30SafetyProjectionError, match="exceeds 1e-06"):
        runtime.advance(sequence, state)
