"""Minimal algorithm interfaces independent of any market environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

from rl_quant.rl.types import ActionBatch, ObservationBatch


RecurrentState = Mapping[str, torch.Tensor]
MetricValue = float | int | torch.Tensor


@runtime_checkable
class Actor(Protocol):
    """A policy that can be used by on-policy, off-policy, or direct algorithms."""

    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: RecurrentState | None = None,
    ) -> ActionBatch: ...


@runtime_checkable
class Critic(Protocol):
    """State-value interface used by advantage-based algorithms."""

    def value(
        self,
        observation: ObservationBatch,
        *,
        recurrent_state: RecurrentState | None = None,
    ) -> torch.Tensor: ...


@runtime_checkable
class ActionValueCritic(Protocol):
    """Action-value interface used by Q-learning and actor-critic algorithms."""

    def q_value(
        self,
        observation: ObservationBatch,
        action: torch.Tensor,
        *,
        recurrent_state: RecurrentState | None = None,
    ) -> torch.Tensor: ...


class Algorithm(ABC):
    """Lifecycle shared by PPO, SAC, DQN, offline RL, and direct optimizers.

    The update payload is intentionally generic: an on-policy implementation can
    require a ``RecurrentSequenceBatch`` while an off-policy implementation can
    accept replay samples.  Concrete algorithms must validate their payload.
    """

    @abstractmethod
    def act(
        self,
        observation: ObservationBatch,
        *,
        deterministic: bool = False,
        recurrent_state: RecurrentState | None = None,
    ) -> ActionBatch:
        """Choose one vectorized action batch."""

    @abstractmethod
    def update(self, batch: Any) -> Mapping[str, MetricValue]:
        """Perform one optimization update and return named diagnostics."""

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return all model, optimizer, scheduler, scaler, and normalization state."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

    def train(self, mode: bool = True) -> Algorithm:
        """Optional mode switch for algorithms backed by torch modules."""

        del mode
        return self

    def eval(self) -> Algorithm:
        return self.train(False)

