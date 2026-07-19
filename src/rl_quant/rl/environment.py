"""Environment protocol shared by domain adapters and RL algorithms."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import torch

from rl_quant.rl.specs import ActionSpec
from rl_quant.rl.types import ActionBatch, ObservationBatch, TransitionBatch


@runtime_checkable
class VectorEnvironment(Protocol):
    """Synchronous vector-environment contract with explicit transitions."""

    action_spec: ActionSpec

    @property
    def batch_size(self) -> int: ...

    def reset(self) -> tuple[ObservationBatch, Mapping[str, torch.Tensor]]: ...

    def step(self, action: ActionBatch | torch.Tensor) -> TransitionBatch: ...

