"""Small stateless PPO actor/critic for bounded adaptive compiler controls."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from rl_quant.rl.algorithm import RecurrentState
from rl_quant.rl.ppo import PPOActorCritic, PPOModelOutput
from rl_quant.rl.types import ObservationBatch
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256

MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1 = 10
MASSIVE_ADAPTIVE_PPO_BIDIRECTIONAL_DIMENSION_V1 = 10
MASSIVE_ADAPTIVE_PPO_POLICY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "observation_dimension": 90,
        "hidden_dimensions": (128, 128),
        "action_dimension": MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1,
        "bidirectional_controls": 10,
        "trade_cost_control": "tanh-normal-on-open-minus-one-to-one",
        "hard_turnover_limit_control": False,
        "deterministic_action": "tanh-normal-mean",
        "duration_semantics": False,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PPO_POLICY_V1_SOURCE_SHA256
        ),
    }
)
MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "model": "massive-adaptive-ppo-actor-critic-v1",
        "observation_dimension": 90,
        "hidden_dimension": 128,
        "observation_key": "adaptive_state",
        "actor_critic_hidden_initialization": "orthogonal-gain-sqrt-two",
        "actor_mean_initialization": "zeros",
        "actor_log_standard_deviation_initialization": -2.0,
        "value_head_initialization": "zeros",
        "seed_source": "experiment-manifest-canonical-seed",
        "rng_scope": "forked-cpu-default-generator-restored",
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PPO_POLICY_V1_SOURCE_SHA256
        ),
    }
)


class MassiveAdaptiveBoundedControlDistributionV1:
    """Ten tanh-Normal controls over the open interval ``(-1, 1)``."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> None:
        if (
            not mean.is_floating_point()
            or mean.shape[-1] != MASSIVE_ADAPTIVE_PPO_BIDIRECTIONAL_DIMENSION_V1
            or log_std.shape != mean.shape
            or log_std.device != mean.device
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(log_std).all().item())
        ):
            raise ValueError("adaptive bounded-control distribution parameters differ")
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 1.0)
        self._normal = torch.distributions.Normal(mean, self.log_std.exp())

    @property
    def leading_shape(self) -> torch.Size:
        return self.mean.shape[:-1]

    def sample(self) -> torch.Tensor:
        return torch.tanh(self._normal.sample())

    def deterministic_action(self) -> torch.Tensor:
        """Return the registered deterministic control for replay/evaluation."""

        return torch.tanh(self.mean)

    def mode(self) -> torch.Tensor:
        """Compatibility alias for the transformed Normal mean action."""

        return self.deterministic_action()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if (
            action.shape != (*self.leading_shape, MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1)
            or action.device != self.mean.device
            or not action.is_floating_point()
            or not bool(torch.isfinite(action).all().item())
            or bool((action.abs() >= 1.0).any().item())
        ):
            raise ValueError("adaptive PPO action lies outside its open support")
        bounded = action
        latent = torch.atanh(bounded)
        correction = torch.log1p(-bounded.square()).sum(dim=-1)
        return self._normal.log_prob(latent).sum(dim=-1) - correction

    def entropy(self) -> torch.Tensor:
        # The tanh transform has no closed-form entropy.  This deterministic
        # base-distribution surrogate is used only as PPO regularization; the
        # exact transformed log probability above remains load bearing.
        return self._normal.entropy().sum(dim=-1)


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
            self.value_head,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

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
        distribution = MassiveAdaptiveBoundedControlDistributionV1(
            mean=mean,
            log_std=log_std,
        )
        value = self.value_head(critic_features).squeeze(-1)
        return PPOModelOutput(distribution=distribution, value=value, recurrent_state={})


def massive_adaptive_ppo_model_state_receipt_v1(
    model: MassiveAdaptivePPOActorCriticV1,
) -> str:
    """Return the canonical tensor identity of one adaptive PPO model."""

    if type(model) is not MassiveAdaptivePPOActorCriticV1:
        raise ValueError("adaptive PPO model type differs")
    rows: list[tuple[str, str, tuple[str, tuple[int, ...], str]]] = []
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        rows.append(
            (
                "str",
                key,
                (str(tensor.dtype), tuple(tensor.shape), digest.hexdigest()),
            )
        )
    return semantic_sha256(tuple(rows))


def build_seeded_massive_adaptive_ppo_model_v1(
    *,
    seed: int,
) -> MassiveAdaptivePPOActorCriticV1:
    """Construct the registered model without consuming ambient RNG state."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("adaptive PPO model initialization seed differs")
    generator = torch.Generator(device="cpu")
    try:
        generator.manual_seed(seed)
    except RuntimeError as error:
        raise ValueError("adaptive PPO model initialization seed differs") from error
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.set_rng_state(generator.get_state())
        model = MassiveAdaptivePPOActorCriticV1(observation_dim=90)
    return model


@lru_cache(maxsize=64, typed=True)
def massive_adaptive_ppo_initial_model_state_receipt_v1(*, seed: int) -> str:
    """Replay the registered seeded initial state once per process and seed."""

    return massive_adaptive_ppo_model_state_receipt_v1(
        build_seeded_massive_adaptive_ppo_model_v1(seed=seed)
    )


__all__ = [
    "MASSIVE_ADAPTIVE_PPO_ACTION_DIMENSION_V1",
    "MASSIVE_ADAPTIVE_PPO_MODEL_INITIALIZATION_V1_SPEC_SHA256",
    "MassiveAdaptiveBoundedControlDistributionV1",
    "MassiveAdaptivePPOActorCriticV1",
    "build_seeded_massive_adaptive_ppo_model_v1",
    "massive_adaptive_ppo_initial_model_state_receipt_v1",
    "massive_adaptive_ppo_model_state_receipt_v1",
]
