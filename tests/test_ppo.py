from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from rl_quant.envs import HistoricalMarketData, VectorPortfolioEnv
from rl_quant.rl import (
    ActionBatch,
    ActionSpec,
    DiagonalNormal,
    MaskedCategorical,
    MaskedDirichlet,
    ObservationBatch,
    OnPolicyTrajectoryBuffer,
    PPOConfig,
    RecurrentActorCritic,
    RecurrentPPO,
    RewardComponents,
    TensorSpec,
    TransitionBatch,
)


def _observation(
    context: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
    episode_start: bool = False,
) -> ObservationBatch:
    return ObservationBatch(
        tensors={"context": context},
        action_mask=action_mask,
        episode_start=torch.full(
            (context.shape[0],), episode_start, dtype=torch.bool, device=context.device
        ),
    )


def _transition(
    observation: ObservationBatch,
    action: ActionBatch,
    reward: torch.Tensor,
    *,
    next_observation: ObservationBatch | None = None,
    terminated: bool = False,
    truncated: bool = False,
    discount: float = 0.99,
) -> TransitionBatch:
    batch_size = reward.shape[0]
    zeros = torch.zeros_like(reward)
    if next_observation is None:
        next_observation = _observation(
            torch.zeros_like(observation.tensors["context"]),
            action_mask=observation.action_mask,
        )
    return TransitionBatch(
        observation=observation,
        action=action,
        executed_action=action.action,
        rewards=RewardComponents(
            gross_return=reward,
            execution_cost=zeros,
            impact_cost=zeros,
            risk_penalty=zeros,
            constraint_penalty=zeros,
            liquidation_cost=zeros,
        ),
        next_observation=next_observation,
        terminated=torch.full((batch_size,), terminated, dtype=torch.bool),
        truncated=torch.full((batch_size,), truncated, dtype=torch.bool),
        discount=torch.full((batch_size,), 0.0 if terminated else discount),
    )


def test_masked_categorical_never_selects_invalid_actions() -> None:
    logits = torch.tensor([[100.0, 0.0, 50.0], [-10.0, 10.0, 0.0]])
    mask = torch.tensor([[False, True, False], [True, False, False]])
    distribution = MaskedCategorical(logits, mask)

    samples = torch.stack([distribution.sample() for _ in range(100)])
    assert samples[:, 0].unique().tolist() == [1]
    assert samples[:, 1].unique().tolist() == [0]
    assert distribution.mode().tolist() == [1, 0]
    assert torch.isfinite(distribution.log_prob(torch.tensor([1, 0]))).all()
    assert torch.isfinite(distribution.entropy()).all()
    with pytest.raises(ValueError, match="invalid"):
        distribution.log_prob(torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="at least one"):
        MaskedCategorical(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_diagonal_normal_and_deterministic_act_are_finite() -> None:
    distribution = DiagonalNormal(
        torch.zeros(2, 2),
        torch.tensor([[100.0, -100.0], [100.0, -100.0]]),
    )
    assert torch.isfinite(distribution.log_prob(torch.zeros(2, 2))).all()
    assert torch.isfinite(distribution.entropy()).all()

    model = RecurrentActorCritic(
        observation_key="context",
        observation_dim=3,
        hidden_dim=8,
        action_dim=2,
        action_kind="normal",
    )
    algorithm = RecurrentPPO(model)
    observation = _observation(torch.randn(4, 3), episode_start=True)
    first = algorithm.act(observation, deterministic=True)
    second = algorithm.act(observation, deterministic=True)
    torch.testing.assert_close(first.action, second.action)
    assert first.action.shape == (4, 2)
    assert torch.isfinite(first.log_prob).all()
    assert torch.isfinite(first.entropy).all()
    assert first.recurrent_state["hidden"].shape == (4, 8)


def test_masked_dirichlet_matches_active_simplex_math_and_has_strict_zeros() -> None:
    concentration = torch.tensor([[2.0, 99.0, 4.0], [8.0, 1.5, 7.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    distribution = MaskedDirichlet(concentration, mask)
    representative = distribution.mode()

    torch.testing.assert_close(
        representative,
        torch.tensor([[1.0 / 3.0, 0.0, 2.0 / 3.0], [0.0, 1.0, 0.0]]),
    )
    reference = torch.distributions.Dirichlet(torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(
        distribution.log_prob(representative)[0],
        reference.log_prob(representative[0, mask[0]]),
    )
    torch.testing.assert_close(distribution.entropy()[0], reference.entropy())
    torch.testing.assert_close(distribution.log_prob(representative)[1], torch.tensor(0.0))
    torch.testing.assert_close(distribution.entropy()[1], torch.tensor(0.0))

    samples = torch.stack([distribution.sample() for _ in range(32)])
    assert torch.equal(samples[..., 1][..., 0], torch.zeros_like(samples[..., 1][..., 0]))
    assert torch.equal(samples[:, 1, [0, 2]], torch.zeros_like(samples[:, 1, [0, 2]]))
    torch.testing.assert_close(samples.sum(dim=-1), torch.ones_like(samples.sum(dim=-1)))
    assert bool((samples[:, 0, [0, 2]] > 0).all())

    # A masked concentration is outside the active event measure and cannot
    # influence its density or entropy.
    changed_masked = concentration.detach().clone()
    changed_masked[0, 1] = 1e4
    changed = MaskedDirichlet(changed_masked, mask)
    torch.testing.assert_close(changed.log_prob(representative), distribution.log_prob(representative))
    torch.testing.assert_close(changed.entropy(), distribution.entropy())

    bad_masked = representative.clone()
    bad_masked[0, 1] = 0.01
    bad_masked[0, 0] -= 0.01
    with pytest.raises(ValueError, match="exactly zero"):
        distribution.log_prob(bad_masked)
    with pytest.raises(ValueError, match="at least one"):
        MaskedDirichlet(torch.ones(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_masked_dirichlet_is_stable_for_reduced_precision_extremes() -> None:
    distribution = MaskedDirichlet(
        torch.tensor([[1e-4, 1e4, 3.0]], dtype=torch.float16),
        torch.tensor([[True, True, False]]),
    )
    representative = distribution.mode()
    assert representative.dtype == torch.float32
    assert representative[0, 0] > 0
    assert representative[0, 2] == 0
    assert torch.isfinite(distribution.log_prob(representative)).all()
    assert torch.isfinite(distribution.entropy()).all()
    sample = distribution.sample()
    assert sample.dtype == torch.float32
    assert sample[0, 0] > 0 and sample[0, 1] > 0 and sample[0, 2] == 0
    assert torch.isfinite(distribution.log_prob(sample)).all()


def test_continuous_ppo_update_handles_recurrent_burn_in_and_padding() -> None:
    torch.manual_seed(4)
    batch_size, horizon = 6, 3
    model = RecurrentActorCritic(
        observation_key="context",
        observation_dim=3,
        hidden_dim=12,
        action_dim=2,
        action_kind="normal",
    )
    algorithm = RecurrentPPO(
        model,
        PPOConfig(learning_rate=1e-3, epochs=2, minibatch_sequences=3),
    )
    buffer = OnPolicyTrajectoryBuffer(horizon=horizon, num_envs=batch_size)
    recurrent_state: dict[str, torch.Tensor] | None = None
    for step in range(horizon):
        observation = _observation(torch.randn(batch_size, 3), episode_start=step == 0)
        state_in = (
            {"hidden": torch.zeros(batch_size, model.hidden_dim)}
            if recurrent_state is None
            else recurrent_state
        )
        action = algorithm.act(observation, recurrent_state=recurrent_state)
        target = observation.tensors["context"][:, :2]
        reward = -(action.action - target).square().sum(dim=-1)
        terminal = step == horizon - 1
        transition = _transition(observation, action, reward, terminated=terminal)
        buffer.add(
            transition,
            value=action.extras["value"],
            next_value=torch.zeros(batch_size),
            recurrent_state=state_in,
        )
        recurrent_state = dict(action.recurrent_state)
    buffer.compute_gae(gae_lambda=0.95)
    sequences = buffer.recurrent_sequences(sequence_length=2, burn_in=1)
    assert (~sequences.valid_mask).any()
    metrics = algorithm.update(sequences)

    for name in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"):
        assert math.isfinite(float(metrics[name])), name
    assert int(metrics["minibatches"]) > 0
    assert algorithm.update_count == 1


def test_dirichlet_ppo_updates_with_masks_and_matches_portfolio_action_spec() -> None:
    torch.manual_seed(12)
    batch_size, horizon, action_dim = 5, 3, 4
    model = RecurrentActorCritic(
        observation_key="context",
        observation_dim=3,
        hidden_dim=12,
        action_dim=action_dim,
        action_kind="dirichlet",
    )
    algorithm = RecurrentPPO(
        model,
        PPOConfig(learning_rate=1e-3, epochs=2, minibatch_sequences=3),
    )
    action_spec = ActionSpec(
        TensorSpec((action_dim,), torch.float32, low=0.0, high=1.0),
        simplex=True,
        cash_index=0,
    )
    buffer = OnPolicyTrajectoryBuffer(horizon=horizon, num_envs=batch_size)
    recurrent_state = algorithm.initial_recurrent_state(
        _observation(torch.zeros(batch_size, 3), episode_start=True)
    )
    for step in range(horizon):
        context = torch.randn(batch_size, 3)
        action_mask = torch.ones((batch_size, action_dim), dtype=torch.bool)
        action_mask[(torch.arange(batch_size) + step) % batch_size, 2] = False
        action_mask[:, 0] = True
        observation = _observation(context, action_mask=action_mask, episode_start=step == 0)
        value = algorithm.value(observation, recurrent_state)
        action = algorithm.act(observation, recurrent_state=recurrent_state)
        action_spec.validate(action.action)
        assert torch.equal(action.action[~action_mask], torch.zeros_like(action.action[~action_mask]))
        torch.testing.assert_close(value, action.extras["value"])
        reward = action.action[:, 1] - 0.25 * action.action[:, 3]
        terminal = step == horizon - 1
        transition = _transition(observation, action, reward, terminated=terminal)
        buffer.add(
            transition,
            value=value,
            next_value=torch.zeros(batch_size),
            recurrent_state=recurrent_state,
        )
        recurrent_state = action.recurrent_state
    buffer.compute_gae(gae_lambda=0.95)
    sequences = buffer.recurrent_sequences(sequence_length=2, burn_in=1)
    metrics = algorithm.update(sequences)
    assert all(
        math.isfinite(float(metrics[name]))
        for name in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm")
    )

    # The same sampled policy output is accepted unchanged by the real
    # portfolio environment's long-only simplex ActionSpec.
    market = HistoricalMarketData(
        features={"context": torch.zeros(batch_size, 2, 3)},
        asset_returns=torch.zeros(batch_size, 1, action_dim),
        availability=torch.ones(batch_size, 2, action_dim, dtype=torch.bool),
    )
    environment = VectorPortfolioEnv(market)
    portfolio_observation, _ = environment.reset()
    portfolio_state = algorithm.initial_recurrent_state(portfolio_observation)
    portfolio_action = algorithm.act(portfolio_observation, recurrent_state=portfolio_state)
    environment.action_spec.validate(portfolio_action.action)


def test_burn_in_reconstructs_state_without_backpropagating_through_history() -> None:
    torch.manual_seed(9)
    model = RecurrentActorCritic(
        observation_key="context",
        observation_dim=3,
        hidden_dim=8,
        action_dim=2,
    )
    context = torch.randn(1, 2, 3, requires_grad=True)
    output = model(
        {"context": context},
        action_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        episode_start=torch.tensor([[True, False]]),
        burn_in=1,
    )
    output.value[:, 1].sum().backward()
    assert context.grad is not None
    torch.testing.assert_close(context.grad[:, 0], torch.zeros_like(context.grad[:, 0]))
    assert context.grad[:, 1].abs().sum().item() > 0


def test_recurrent_ppo_learns_a_planted_contextual_bandit_and_restores_checkpoint() -> None:
    torch.manual_seed(7)
    batch_size = 64
    model = RecurrentActorCritic(
        observation_key="context",
        observation_dim=2,
        hidden_dim=16,
        action_dim=2,
        action_kind="categorical",
    )
    algorithm = RecurrentPPO(
        model,
        PPOConfig(
            learning_rate=5e-3,
            epochs=4,
            minibatch_sequences=32,
            entropy_coefficient=0.01,
            value_coefficient=0.25,
        ),
    )
    all_actions_available = torch.ones((batch_size, 2), dtype=torch.bool)

    for update in range(30):
        labels = (torch.arange(batch_size) + update) % 2
        context = F.one_hot(labels, num_classes=2).float()
        permutation = torch.randperm(batch_size)
        labels = labels[permutation]
        context = context[permutation]
        observation = _observation(
            context,
            action_mask=all_actions_available,
            episode_start=True,
        )
        action = algorithm.act(observation)
        reward = torch.where(action.action == labels, 1.0, -1.0)
        transition = _transition(observation, action, reward, terminated=True)
        buffer = OnPolicyTrajectoryBuffer(horizon=1, num_envs=batch_size)
        buffer.add(
            transition,
            value=action.extras["value"],
            next_value=torch.zeros(batch_size),
            recurrent_state={"hidden": torch.zeros(batch_size, model.hidden_dim)},
        )
        buffer.compute_gae(gae_lambda=1.0)
        metrics = algorithm.update(buffer.recurrent_sequences(sequence_length=1))
        assert all(
            math.isfinite(float(metrics[name]))
            for name in ("loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        )

    evaluation = _observation(
        torch.eye(2),
        action_mask=torch.ones((2, 2), dtype=torch.bool),
        episode_start=True,
    )
    learned = algorithm.act(evaluation, deterministic=True)
    assert learned.action.tolist() == [0, 1]

    checkpoint = copy.deepcopy(algorithm.state_dict())
    restored = RecurrentPPO(
        RecurrentActorCritic(
            observation_key="context",
            observation_dim=2,
            hidden_dim=16,
            action_dim=2,
            action_kind="categorical",
        ),
        algorithm.config,
    )
    restored.load_state_dict(checkpoint)
    assert restored.update_count == algorithm.update_count
    restored_action = restored.act(evaluation, deterministic=True)
    torch.testing.assert_close(restored_action.action, learned.action)

    resume_batch = buffer.recurrent_sequences(sequence_length=1)
    algorithm.update(resume_batch)
    restored.update(resume_batch)
    for name, value in algorithm.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name])


def test_ppo_rejects_malformed_rng_checkpoint_before_mutating_model() -> None:
    algorithm = RecurrentPPO(
        RecurrentActorCritic(
            observation_key="context",
            observation_dim=2,
            hidden_dim=4,
            action_dim=2,
            action_kind="categorical",
        ),
        PPOConfig(epochs=1, minibatch_sequences=1),
    )
    checkpoint = copy.deepcopy(algorithm.state_dict())
    checkpoint["minibatch_rng_state"] = torch.zeros((2, 2), dtype=torch.uint8)
    restored = RecurrentPPO(
        RecurrentActorCritic(
            observation_key="context",
            observation_dim=2,
            hidden_dim=4,
            action_dim=2,
            action_kind="categorical",
        ),
        algorithm.config,
    )
    before = {name: value.detach().clone() for name, value in restored.model.state_dict().items()}

    with pytest.raises(ValueError, match="CPU uint8"):
        restored.load_state_dict(checkpoint)

    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
