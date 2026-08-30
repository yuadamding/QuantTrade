"""Small stateless PPO actor/critic for bounded adaptive compiler controls."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from rl_quant.rl.algorithm import RecurrentState
from rl_quant.rl.ppo import PPOActorCritic, PPOModelOutput
from rl_quant.rl.types import ObservationBatch
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256

MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1 = 10
MASSIVE_ADAPTIVE_PPO_BIDIRECTIONAL_DIMENSION_V1 = 9
MASSIVE_ADAPTIVE_PPO_POLICY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "observation_dimension": 90,
        "hidden_dimensions": (128, 128),
        "action_dimension": MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1,
        "bidirectional_controls": 9,
        "turnover_control": "beta-on-open-unit-interval",
        "deterministic_action": "tanh-normal-mean-and-beta-mean",
        "duration_semantics": False,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PPO_POLICY_V1_SOURCE_SHA256
        ),
    }
)


class MassiveAdaptiveBoundedControlDistributionV1:
    """Nine tanh-Normal controls and one Beta turnover-tightening control."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        turnover_alpha: torch.Tensor,
        turnover_beta: torch.Tensor,
    ) -> None:
        if (
            not mean.is_floating_point()
            or mean.shape[-1] != MASSIVE_ADAPTIVE_PPO_BIDIRECTIONAL_DIMENSION_V1
            or log_std.shape != mean.shape
            or turnover_alpha.shape != mean.shape[:-1]
            or turnover_beta.shape != mean.shape[:-1]
            or any(
                value.device != mean.device
                for value in (log_std, turnover_alpha, turnover_beta)
            )
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(log_std).all().item())
            or not bool(torch.isfinite(turnover_alpha).all().item())
            or not bool(torch.isfinite(turnover_beta).all().item())
            or bool((turnover_alpha <= 0.0).any().item())
            or bool((turnover_beta <= 0.0).any().item())
        ):
            raise ValueError("adaptive bounded-control distribution parameters differ")
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 1.0)
        self.turnover_alpha = turnover_alpha
        self.turnover_beta = turnover_beta
        self._normal = torch.distributions.Normal(mean, self.log_std.exp())
        self._turnover = torch.distributions.Beta(turnover_alpha, turnover_beta)

    @property
    def leading_shape(self) -> torch.Size:
        return self.mean.shape[:-1]

    def sample(self) -> torch.Tensor:
        bidirectional = torch.tanh(self._normal.sample())
        turnover = self._turnover.sample().unsqueeze(-1)
        return torch.cat((bidirectional, turnover), dim=-1)

    def deterministic_action(self) -> torch.Tensor:
        """Return the registered deterministic control for replay/evaluation."""

        bidirectional = torch.tanh(self.mean)
        turnover = (
            self.turnover_alpha / (self.turnover_alpha + self.turnover_beta)
        ).unsqueeze(-1)
        return torch.cat((bidirectional, turnover), dim=-1)

    def mode(self) -> torch.Tensor:
        """Compatibility alias; the Beta component uses its mean, not its mode."""

        return self.deterministic_action()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if (
            action.shape != (*self.leading_shape, MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1)
            or action.device != self.mean.device
            or not action.is_floating_point()
            or not bool(torch.isfinite(action).all().item())
            or bool((action[..., :9].abs() >= 1.0).any().item())
            or bool((action[..., 9] <= 0.0).any().item())
            or bool((action[..., 9] >= 1.0).any().item())
        ):
            raise ValueError("adaptive PPO action lies outside its open support")
        bounded = action[..., :9]
        latent = torch.atanh(bounded)
        correction = torch.log1p(-bounded.square()).sum(dim=-1)
        normal_log_prob = self._normal.log_prob(latent).sum(dim=-1) - correction
        return normal_log_prob + self._turnover.log_prob(action[..., 9])

    def entropy(self) -> torch.Tensor:
        # The tanh transform has no closed-form entropy.  This deterministic
        # base-distribution surrogate is used only as PPO regularization; the
        # exact transformed log probability above remains load bearing.
        return self._normal.entropy().sum(dim=-1) + self._turnover.entropy()


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
    )


class MassiveAdaptivePPOActorCriticV1(PPOActorCritic):
    """Separate two-layer actor and critic initialized near neutral control."""

    def __init__(
        self,
        *,
        observation_dim: int,
        hidden_dim: int = 128,
        observation_key: str = "adaptive_state",
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or hidden_dim <= 0 or not observation_key:
            raise ValueError("adaptive PPO model dimensions are invalid")
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.observation_key = observation_key
        self.actor = _mlp(observation_dim, hidden_dim)
        self.critic = _mlp(observation_dim, hidden_dim)
        self.actor_mean = nn.Linear(
            hidden_dim, MASSIVE_ADAPTIVE_PPO_BIDIRECTIONAL_DIMENSION_V1
        )
        self.actor_log_std = nn.Parameter(
            torch.full((MASSIVE_ADAPTIVE_PPO_BIDIRECTIONAL_DIMENSION_V1,), -2.0)
        )
        self.turnover_alpha = nn.Linear(hidden_dim, 1)
        self.turnover_beta = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for network in (self.actor, self.critic):
            for module in network:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(module.bias)
        for module in (
            self.actor_mean,
            self.turnover_alpha,
            self.turnover_beta,
            self.value_head,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.turnover_alpha.bias.data.fill_(-2.5)
        self.turnover_beta.bias.data.fill_(2.5)

    def initial_recurrent_state(
        self, observation: ObservationBatch
    ) -> Mapping[str, torch.Tensor]:
        del observation
        return {}

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        action_mask: torch.Tensor | None = None,
        recurrent_state: RecurrentState | None = None,
        episode_start: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        burn_in: int = 0,
    ) -> PPOModelOutput:
        del episode_start, valid_mask
        if action_mask is not None or recurrent_state:
            raise ValueError("adaptive PPO policy is stateless and unmasked")
        if burn_in != 0:
            raise ValueError("stateless adaptive PPO cannot consume burn-in")
        try:
            inputs = observations[self.observation_key]
        except KeyError as exc:
            raise ValueError("adaptive PPO observation field is absent") from exc
        if (
            not inputs.is_floating_point()
            or inputs.ndim not in (2, 3)
            or inputs.shape[-1] != self.observation_dim
            or not bool(torch.isfinite(inputs).all().item())
        ):
            raise ValueError("adaptive PPO observation tensor differs")
        actor_features = self.actor(inputs)
        critic_features = self.critic(inputs)
        mean = self.actor_mean(actor_features)
        log_std = self.actor_log_std.expand_as(mean)
        alpha = F.softplus(self.turnover_alpha(actor_features).squeeze(-1)) + 0.05
        beta = F.softplus(self.turnover_beta(actor_features).squeeze(-1)) + 0.05
        distribution = MassiveAdaptiveBoundedControlDistributionV1(
            mean=mean,
            log_std=log_std,
            turnover_alpha=alpha,
            turnover_beta=beta,
        )
        value = self.value_head(critic_features).squeeze(-1)
        return PPOModelOutput(distribution=distribution, value=value, recurrent_state={})


__all__ = [
    "MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1",
    "MassiveAdaptiveBoundedControlDistributionV1",
    "MassiveAdaptivePPOActorCriticV1",
]
