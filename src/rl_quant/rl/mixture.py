"""Domain-neutral regime routing and same-support action mixtures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
from torch import nn

from rl_quant.rl.ppo import DiagonalNormal, MaskedCategorical, MaskedDirichlet


@dataclass(frozen=True)
class RouterOutput:
    """Validated expert routing probabilities for arbitrary leading dimensions."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    expert_mask: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.probabilities.shape
        if len(shape) < 1 or shape[-1] < 1:
            raise ValueError("Router probabilities need shape [..., expert] with at least one expert.")
        if (
            self.logits.shape != shape
            or self.log_probabilities.shape != shape
            or self.expert_mask.shape != shape
        ):
            raise ValueError("All router tensors must have identical shapes.")
        if not self.probabilities.is_floating_point() or not self.logits.is_floating_point():
            raise ValueError("Router logits and probabilities must be floating point.")
        if self.log_probabilities.dtype != self.probabilities.dtype:
            raise ValueError("Router probabilities and log probabilities must share a dtype.")
        if self.expert_mask.dtype != torch.bool:
            raise ValueError("expert_mask must be torch.bool.")
        device = self.probabilities.device
        if any(value.device != device for value in (self.logits, self.log_probabilities, self.expert_mask)):
            raise ValueError("All router tensors must share one device.")
        if bool((~self.expert_mask.any(dim=-1)).any().item()):
            raise ValueError("Every router row must enable at least one expert.")
        if not bool(torch.isfinite(self.probabilities).all().item()) or bool(
            (self.probabilities < 0).any().item()
        ):
            raise ValueError("Router probabilities must be finite and nonnegative.")
        if bool((self.probabilities[~self.expert_mask] != 0).any().item()):
            raise ValueError("Masked experts must have exactly zero routing probability.")
        sums = self.probabilities.sum(dim=-1)
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0)):
            raise ValueError("Router probabilities must sum to one.")
        active_logs = self.log_probabilities[self.expert_mask]
        if not bool(torch.isfinite(active_logs).all().item()):
            raise ValueError("Active router log probabilities must be finite.")
        if bool((~torch.isneginf(self.log_probabilities[~self.expert_mask])).any().item()):
            raise ValueError("Masked router log probabilities must be -inf.")
        if not bool(
            torch.allclose(
                active_logs.exp(),
                self.probabilities[self.expert_mask],
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise ValueError("Router probabilities and log probabilities disagree.")

    @property
    def num_experts(self) -> int:
        return self.probabilities.shape[-1]

    @property
    def leading_shape(self) -> torch.Size:
        return self.probabilities.shape[:-1]

    @property
    def weights(self) -> torch.Tensor:
        """Alias emphasizing that probabilities are mixture weights."""

        return self.probabilities

    @property
    def entropy(self) -> torch.Tensor:
        safe_log_probabilities = torch.where(
            self.probabilities > 0,
            self.log_probabilities,
            torch.zeros_like(self.log_probabilities),
        )
        return -(self.probabilities * safe_log_probabilities).sum(dim=-1)

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        *,
        expert_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> RouterOutput:
        if not logits.is_floating_point() or logits.ndim < 1:
            raise ValueError("Router logits need floating shape [..., expert].")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("Router logits must be finite.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Router temperature must be finite and positive.")
        if expert_mask is None:
            expert_mask = torch.ones_like(logits, dtype=torch.bool)
        if expert_mask.shape != logits.shape or expert_mask.dtype != torch.bool or expert_mask.device != logits.device:
            raise ValueError("expert_mask must be bool and exactly match router logits shape/device.")
        if bool((~expert_mask.any(dim=-1)).any().item()):
            raise ValueError("Every router row must enable at least one expert.")
        scaled = logits / temperature
        effective_logits = torch.where(
            expert_mask,
            scaled,
            torch.full_like(scaled, -torch.inf),
        )
        log_probabilities = torch.log_softmax(effective_logits, dim=-1)
        log_probabilities = torch.where(
            expert_mask,
            log_probabilities,
            torch.full_like(log_probabilities, -torch.inf),
        )
        probabilities = torch.where(
            expert_mask,
            log_probabilities.exp(),
            torch.zeros_like(log_probabilities),
        )
        return cls(effective_logits, probabilities, log_probabilities, expert_mask)

    @classmethod
    def from_probabilities(
        cls,
        probabilities: torch.Tensor,
        *,
        expert_mask: torch.Tensor | None = None,
    ) -> RouterOutput:
        if not probabilities.is_floating_point() or probabilities.ndim < 1:
            raise ValueError("Router probabilities need floating shape [..., expert].")
        if expert_mask is None:
            expert_mask = probabilities > 0
        if (
            expert_mask.shape != probabilities.shape
            or expert_mask.dtype != torch.bool
            or expert_mask.device != probabilities.device
        ):
            raise ValueError("expert_mask must be bool and exactly match router probabilities shape/device.")
        safe_probabilities = torch.where(
            probabilities > 0,
            probabilities,
            torch.ones_like(probabilities),
        )
        log_probabilities = torch.where(
            probabilities > 0,
            safe_probabilities.log(),
            torch.full_like(probabilities, -torch.inf),
        )
        return cls(log_probabilities, probabilities, log_probabilities, expert_mask)


class RegimeRouter(nn.Module):
    """Small learnable router mapping domain features to expert probabilities."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_experts <= 0:
            raise ValueError("input_dim and num_experts must be positive.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive or None.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        if hidden_dim is None:
            self.network: nn.Module = nn.Linear(input_dim, num_experts)
            nn.init.zeros_(self.network.weight)
            nn.init.zeros_(self.network.bias)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_experts),
            )
            output = self.network[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        features: torch.Tensor,
        *,
        expert_mask: torch.Tensor | None = None,
    ) -> RouterOutput:
        if not features.is_floating_point() or features.shape[-1] != self.input_dim:
            raise ValueError(f"Router features need floating final dimension {self.input_dim}.")
        logits = self.network(features)
        return RouterOutput.from_logits(
            logits,
            expert_mask=expert_mask,
            temperature=self.temperature,
        )


SupportedActionDistribution = MaskedCategorical | DiagonalNormal | MaskedDirichlet


@dataclass(frozen=True)
class RoutedAction:
    action: torch.Tensor
    expert_index: torch.Tensor


class MixtureActionDistribution:
    """Exact marginal action distribution for same-support regime experts.

    Continuous/simplex mixture entropy has no general closed form. In that
    case :meth:`entropy` deliberately returns the clearly labeled
    ``entropy_upper_bound``: router entropy plus routed component entropy.
    Categorical mixtures expose their exact marginal entropy.
    """

    def __init__(
        self,
        components: Sequence[SupportedActionDistribution],
        router: RouterOutput,
    ) -> None:
        if not components:
            raise ValueError("A mixture requires at least one component expert.")
        self.components = tuple(components)
        self.router = router
        if len(self.components) != router.num_experts:
            raise ValueError(
                f"Router has {router.num_experts} experts but {len(self.components)} components were supplied."
            )
        component_type = type(self.components[0])
        if component_type not in (MaskedCategorical, DiagonalNormal, MaskedDirichlet):
            raise TypeError(f"Unsupported mixture component type {component_type.__name__}.")
        if any(type(component) is not component_type for component in self.components):
            raise ValueError("All mixture components must have the same distribution type and support.")
        first = self.components[0]
        if first.leading_shape != router.leading_shape:
            raise ValueError(
                f"Component leading shape {tuple(first.leading_shape)} does not match router "
                f"{tuple(router.leading_shape)}."
            )
        if any(component.leading_shape != first.leading_shape for component in self.components[1:]):
            raise ValueError("All mixture components must share one leading shape.")
        component_device = self._component_device(first)
        if router.probabilities.device != component_device:
            raise ValueError("Router and component distributions must share one device.")

        if isinstance(first, MaskedCategorical):
            action_shape = first.logits.shape
            first_mask = first.mask
            for component in self.components[1:]:
                assert isinstance(component, MaskedCategorical)
                if component.logits.shape != action_shape or not torch.equal(component.mask, first_mask):
                    raise ValueError("Categorical experts must use identical action shapes and masks.")
            self._kind: Literal["categorical", "continuous", "simplex"] = "categorical"
            self._action_mask: torch.Tensor | None = first_mask
        elif isinstance(first, DiagonalNormal):
            action_shape = first.mean.shape
            for component in self.components[1:]:
                assert isinstance(component, DiagonalNormal)
                if component.mean.shape != action_shape:
                    raise ValueError("Normal experts must use identical event shapes.")
            self._kind = "continuous"
            self._action_mask = None
        else:
            assert isinstance(first, MaskedDirichlet)
            action_shape = first.concentration.shape
            first_mask = first.mask
            for component in self.components[1:]:
                assert isinstance(component, MaskedDirichlet)
                if component.concentration.shape != action_shape or not torch.equal(component.mask, first_mask):
                    raise ValueError("Dirichlet experts must use identical action shapes and masks.")
            self._kind = "simplex"
            self._action_mask = first_mask

    @staticmethod
    def _component_device(component: SupportedActionDistribution) -> torch.device:
        if isinstance(component, MaskedCategorical):
            return component.logits.device
        if isinstance(component, DiagonalNormal):
            return component.mean.device
        return component.concentration.device

    @property
    def leading_shape(self) -> torch.Size:
        return self.router.leading_shape

    @property
    def router_probabilities(self) -> torch.Tensor:
        return self.router.probabilities

    @property
    def router_weights(self) -> torch.Tensor:
        return self.router.weights

    @property
    def router_log_probabilities(self) -> torch.Tensor:
        return self.router.log_probabilities

    @property
    def expert_mask(self) -> torch.Tensor:
        return self.router.expert_mask

    @property
    def action_mask(self) -> torch.Tensor | None:
        return self._action_mask

    @property
    def entropy_kind(self) -> Literal["exact", "upper_bound"]:
        return "exact" if self._kind == "categorical" else "upper_bound"

    @property
    def action_kind(self) -> Literal["categorical", "continuous", "simplex"]:
        """The common support shared by all specialist distributions."""

        return self._kind

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        component_log_probs = torch.stack(
            [component.log_prob(action) for component in self.components],
            dim=-1,
        )
        return torch.logsumexp(
            self.router.log_probabilities + component_log_probs,
            dim=-1,
        )

    def _select_expert_values(
        self,
        values: Sequence[torch.Tensor],
        expert_index: torch.Tensor,
    ) -> torch.Tensor:
        first = values[0]
        leading_ndim = len(self.leading_shape)
        event_shape = first.shape[leading_ndim:]
        if any(value.shape != first.shape for value in values[1:]):
            raise ValueError("Experts returned incompatible action shapes.")
        expert_axis = leading_ndim
        stacked = torch.stack(tuple(values), dim=expert_axis)
        if not event_shape:
            return stacked.gather(expert_axis, expert_index.unsqueeze(-1)).squeeze(expert_axis)
        index = expert_index.reshape(*self.leading_shape, 1, *(1 for _ in event_shape))
        index = index.expand(*self.leading_shape, 1, *event_shape)
        return stacked.gather(expert_axis, index).squeeze(expert_axis)

    def sample_with_expert(self) -> RoutedAction:
        expert_index = torch.distributions.Categorical(probs=self.router.probabilities).sample()
        action = self._select_expert_values(
            [component.sample() for component in self.components],
            expert_index,
        )
        return RoutedAction(action=action, expert_index=expert_index)

    def sample(self) -> torch.Tensor:
        return self.sample_with_expert().action

    def mode(self) -> torch.Tensor:
        """Return a coherent deterministic action for the mixture.

        Categorical mixtures have an exact marginal distribution, so this is
        its MAP action. A continuous/simplex mixture's mathematical mode has no
        general closed form; there we select the highest-probability router
        expert's deterministic representative instead of a low-density weighted
        compromise never sampled by any expert.
        """

        if self._kind == "categorical":
            categorical_probabilities = []
            for component in self.components:
                assert isinstance(component, MaskedCategorical)
                masked_logits = component.logits.masked_fill(
                    ~component.mask,
                    torch.finfo(component.logits.dtype).min,
                )
                categorical_probabilities.append(torch.softmax(masked_logits, dim=-1))
            stacked = torch.stack(categorical_probabilities, dim=len(self.leading_shape))
            marginal = (
                stacked * self.router.probabilities.unsqueeze(-1)
            ).sum(dim=len(self.leading_shape))
            return marginal.argmax(dim=-1)
        modes = [component.mode() for component in self.components]
        expert_index = self.router.probabilities.argmax(dim=-1)
        return self._select_expert_values(modes, expert_index)

    def entropy_upper_bound(self) -> torch.Tensor:
        component_entropy = torch.stack(
            [component.entropy() for component in self.components],
            dim=-1,
        )
        return self.router.entropy + (self.router.probabilities * component_entropy).sum(dim=-1)

    def exact_entropy(self) -> torch.Tensor:
        if self._kind != "categorical":
            raise NotImplementedError(
                "Exact entropy is unavailable for continuous/simplex mixtures; "
                "use entropy_upper_bound()."
            )
        categorical_probabilities = []
        for component in self.components:
            assert isinstance(component, MaskedCategorical)
            masked_logits = component.logits.masked_fill(
                ~component.mask,
                torch.finfo(component.logits.dtype).min,
            )
            categorical_probabilities.append(torch.softmax(masked_logits, dim=-1))
        stacked = torch.stack(categorical_probabilities, dim=len(self.leading_shape))
        mixture_probabilities = (
            stacked * self.router.probabilities.unsqueeze(-1)
        ).sum(dim=len(self.leading_shape))
        safe_probabilities = torch.where(
            mixture_probabilities > 0,
            mixture_probabilities,
            torch.ones_like(mixture_probabilities),
        )
        return -(mixture_probabilities * safe_probabilities.log()).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Exact categorical entropy, otherwise the documented upper bound."""

        return self.exact_entropy() if self._kind == "categorical" else self.entropy_upper_bound()
