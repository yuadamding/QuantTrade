from __future__ import annotations

import math

import pytest
import torch

from rl_quant.rl import (
    ActionBatch,
    DiagonalNormal,
    MaskedCategorical,
    MaskedDirichlet,
    MixtureActionDistribution,
    PPOModelOutput,
    RegimeRouter,
    RouterOutput,
)


def test_router_probabilities_weights_and_expert_mask_are_explicit() -> None:
    logits = torch.tensor([[0.0, 100.0, 1.0], [2.0, -3.0, 4.0]], requires_grad=True)
    expert_mask = torch.tensor([[True, False, True], [True, True, False]])
    router = RouterOutput.from_logits(logits, expert_mask=expert_mask, temperature=0.5)

    assert torch.equal(router.probabilities[~expert_mask], torch.zeros(2))
    assert torch.isneginf(router.log_probabilities[~expert_mask]).all()
    torch.testing.assert_close(router.probabilities.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(router.weights, router.probabilities)
    assert router.entropy.shape == (2,)

    objective = (router.probabilities * torch.tensor([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])).sum()
    objective.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[~expert_mask], torch.zeros_like(logits.grad[~expert_mask]))

    learnable = RegimeRouter(input_dim=4, num_experts=3)
    output = learnable(torch.randn(2, 5, 4), expert_mask=torch.ones(2, 5, 3, dtype=torch.bool))
    assert output.probabilities.shape == (2, 5, 3)
    torch.testing.assert_close(output.probabilities, torch.full((2, 5, 3), 1.0 / 3.0))


def test_exact_mixture_log_prob_uses_stable_logsumexp_and_backpropagates() -> None:
    first_mean = torch.tensor([[0.0], [1.0]], requires_grad=True)
    second_mean = torch.tensor([[2.0], [-1.0]], requires_grad=True)
    zero_log_std = torch.zeros(2, 1)
    components = (
        DiagonalNormal(first_mean, zero_log_std),
        DiagonalNormal(second_mean, zero_log_std),
    )
    router_logits = torch.tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
    router = RouterOutput.from_logits(router_logits)
    mixture = MixtureActionDistribution(components, router)
    action = torch.tensor([[0.5], [-0.25]])

    component_log_prob = torch.stack([component.log_prob(action) for component in components], dim=-1)
    expected = torch.logsumexp(router.log_probabilities + component_log_prob, dim=-1)
    torch.testing.assert_close(mixture.log_prob(action), expected)

    loss = -mixture.log_prob(action).mean()
    loss.backward()
    assert router_logits.grad is not None and torch.isfinite(router_logits.grad).all()
    assert first_mean.grad is not None and torch.isfinite(first_mean.grad).all()
    assert second_mean.grad is not None and torch.isfinite(second_mean.grad).all()


def test_sampling_records_the_routed_expert() -> None:
    components = (
        DiagonalNormal(torch.tensor([[0.0], [0.0]]), torch.full((2, 1), -20.0)),
        DiagonalNormal(torch.tensor([[10.0], [10.0]]), torch.full((2, 1), -20.0)),
    )
    router = RouterOutput.from_probabilities(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    mixture = MixtureActionDistribution(components, router)
    routed = mixture.sample_with_expert()

    assert routed.expert_index.tolist() == [0, 1]
    torch.testing.assert_close(routed.action, torch.tensor([[0.0], [10.0]]), atol=1e-6, rtol=0.0)


def test_deterministic_actions_preserve_specialist_semantics() -> None:
    categorical_components = (
        MaskedCategorical(torch.tensor([[5.0, 0.0], [5.0, 0.0]])),
        MaskedCategorical(torch.tensor([[0.0, 5.0], [0.0, 5.0]])),
    )
    categorical_router = RouterOutput.from_probabilities(torch.tensor([[0.4, 0.6], [0.8, 0.2]]))
    categorical = MixtureActionDistribution(categorical_components, categorical_router)
    # Discrete decisions use exact marginal MAP.
    assert categorical.mode().tolist() == [1, 0]

    marginal_map = MixtureActionDistribution(
        (
            MaskedCategorical(torch.log(torch.tensor([[0.51, 0.49]]))),
            MaskedCategorical(torch.log(torch.tensor([[0.001, 0.999]]))),
        ),
        RouterOutput.from_probabilities(torch.tensor([[0.51, 0.49]])),
    )
    assert marginal_map.mode().item() == 1

    normal_components = (
        DiagonalNormal(torch.tensor([[0.0, 2.0]]), torch.zeros(1, 2)),
        DiagonalNormal(torch.tensor([[10.0, 6.0]]), torch.zeros(1, 2)),
    )
    normal_router = RouterOutput.from_probabilities(torch.tensor([[0.25, 0.75]]))
    normal = MixtureActionDistribution(normal_components, normal_router)
    torch.testing.assert_close(normal.mode(), torch.tensor([[10.0, 6.0]]))

    action_mask = torch.tensor([[True, True, False]])
    simplex_components = (
        MaskedDirichlet(torch.tensor([[2.0, 1.0, 50.0]]), action_mask),
        MaskedDirichlet(torch.tensor([[1.0, 3.0, 0.1]]), action_mask),
    )
    simplex_router = RouterOutput.from_probabilities(torch.tensor([[0.25, 0.75]]))
    simplex = MixtureActionDistribution(simplex_components, simplex_router)
    torch.testing.assert_close(simplex.mode(), simplex_components[1].mode())
    torch.testing.assert_close(simplex.mode().sum(dim=-1), torch.ones(1))
    assert simplex.mode()[0, 2] == 0
    assert torch.equal(simplex.action_mask, action_mask)


def test_same_support_and_action_masks_are_required() -> None:
    router = RouterOutput.from_probabilities(torch.tensor([[0.5, 0.5]]))
    first_mask = torch.tensor([[True, False, True]])
    second_mask = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="identical action shapes and masks"):
        MixtureActionDistribution(
            (
                MaskedCategorical(torch.zeros(1, 3), first_mask),
                MaskedCategorical(torch.zeros(1, 3), second_mask),
            ),
            router,
        )
    with pytest.raises(ValueError, match="same distribution type"):
        MixtureActionDistribution(
            (
                MaskedCategorical(torch.zeros(1, 2)),
                DiagonalNormal(torch.zeros(1, 2), torch.zeros(1, 2)),
            ),
            router,
        )
    with pytest.raises(ValueError, match="identical action shapes and masks"):
        MixtureActionDistribution(
            (
                MaskedDirichlet(torch.ones(1, 3), first_mask),
                MaskedDirichlet(torch.ones(1, 3), second_mask),
            ),
            router,
        )
    with pytest.raises(ValueError, match="at least one expert"):
        RouterOutput.from_logits(torch.zeros(1, 2), expert_mask=torch.zeros(1, 2, dtype=torch.bool))


def test_entropy_is_exact_for_categorical_and_labeled_upper_bound_otherwise() -> None:
    categorical_components = (
        MaskedCategorical(torch.tensor([[math.log(0.9), math.log(0.1)]])),
        MaskedCategorical(torch.tensor([[math.log(0.2), math.log(0.8)]])),
    )
    router = RouterOutput.from_probabilities(torch.tensor([[0.25, 0.75]]))
    categorical = MixtureActionDistribution(categorical_components, router)
    mixture_probabilities = torch.tensor([[0.25 * 0.9 + 0.75 * 0.2, 0.25 * 0.1 + 0.75 * 0.8]])
    expected_entropy = -(mixture_probabilities * mixture_probabilities.log()).sum(dim=-1)

    assert categorical.entropy_kind == "exact"
    torch.testing.assert_close(categorical.exact_entropy(), expected_entropy)
    torch.testing.assert_close(categorical.entropy(), expected_entropy)
    assert bool((categorical.entropy_upper_bound() >= categorical.entropy() - 1e-6).all())

    continuous = MixtureActionDistribution(
        (
            DiagonalNormal(torch.tensor([[0.0]]), torch.zeros(1, 1)),
            DiagonalNormal(torch.tensor([[2.0]]), torch.zeros(1, 1)),
        ),
        router,
    )
    assert continuous.entropy_kind == "upper_bound"
    torch.testing.assert_close(continuous.entropy(), continuous.entropy_upper_bound())
    assert torch.isfinite(continuous.entropy()).all()
    with pytest.raises(NotImplementedError, match="entropy_upper_bound"):
        continuous.exact_entropy()

    masked_logits = torch.tensor([[1.0, 20.0, 3.0]], requires_grad=True)
    action_mask = torch.tensor([[True, False, True]])
    masked = MixtureActionDistribution(
        (
            MaskedCategorical(masked_logits, action_mask),
            MaskedCategorical(masked_logits + 0.5, action_mask),
        ),
        RouterOutput.from_probabilities(torch.tensor([[0.5, 0.5]])),
    )
    masked.entropy().sum().backward()
    assert masked_logits.grad is not None and torch.isfinite(masked_logits.grad).all()


def test_mixture_satisfies_existing_ppo_distribution_contract() -> None:
    components = (
        MaskedDirichlet(torch.tensor([[2.0, 1.0, 3.0]])),
        MaskedDirichlet(torch.tensor([[1.0, 4.0, 2.0]])),
    )
    mixture = MixtureActionDistribution(
        components,
        RouterOutput.from_probabilities(torch.tensor([[0.3, 0.7]])),
    )
    output = PPOModelOutput(
        distribution=mixture,
        value=torch.zeros(1),
        recurrent_state={},
    )
    action = output.distribution.sample()
    batch = ActionBatch(
        action=action,
        log_prob=output.distribution.log_prob(action),
        entropy=output.distribution.entropy(),
    )
    torch.testing.assert_close(batch.action.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(batch.log_prob).all()
    assert torch.isfinite(batch.entropy).all()
