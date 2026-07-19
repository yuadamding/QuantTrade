"""Runtime tensor and action specifications for algorithm-neutral RL components.

The specifications deliberately describe tensors rather than models.  Environments,
policies, replay stores, and evaluators can therefore agree on shapes and dtypes
without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ShapeDim = int | None
ActionKind = Literal["continuous", "discrete", "hybrid"]


@dataclass(frozen=True)
class TensorSpec:
    """Describe the non-leading dimensions, dtype, and optional numeric bounds.

    ``None`` is a wildcard dimension.  ``leading_dims`` in :meth:`validate`
    separates batch/time dimensions from the declared event shape.
    """

    shape: tuple[ShapeDim, ...]
    dtype: torch.dtype
    low: float | None = None
    high: float | None = None
    finite: bool = True

    def __post_init__(self) -> None:
        for dim in self.shape:
            if dim is not None and dim < 0:
                raise ValueError(f"TensorSpec dimensions must be nonnegative or None; got {self.shape}.")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"TensorSpec low ({self.low}) exceeds high ({self.high}).")

    def validate(self, value: torch.Tensor, *, name: str = "tensor", leading_dims: int = 1) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor; got {type(value).__name__}.")
        if leading_dims < 0:
            raise ValueError(f"leading_dims must be nonnegative; got {leading_dims}.")
        expected_ndim = leading_dims + len(self.shape)
        if value.ndim != expected_ndim:
            raise ValueError(f"{name} must have {expected_ndim} dimensions; got shape {tuple(value.shape)}.")
        actual_event = value.shape[leading_dims:]
        for index, (actual, expected) in enumerate(zip(actual_event, self.shape, strict=True)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} event dimension {index} must be {expected}; got {actual} in {tuple(value.shape)}."
                )
        if value.dtype != self.dtype:
            raise ValueError(f"{name} must have dtype {self.dtype}; got {value.dtype}.")
        if self.finite and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values.")
        if self.low is not None and bool((value < self.low).any().item()):
            raise ValueError(f"{name} contains values below {self.low}.")
        if self.high is not None and bool((value > self.high).any().item()):
            raise ValueError(f"{name} contains values above {self.high}.")

    def zeros(
        self,
        leading_shape: tuple[int, ...],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if any(dim is None for dim in self.shape):
            raise ValueError("Cannot allocate from a TensorSpec containing wildcard dimensions.")
        event_shape = tuple(dim for dim in self.shape if dim is not None)
        return torch.zeros((*leading_shape, *event_shape), dtype=self.dtype, device=device)


@dataclass(frozen=True)
class ActionSpec:
    """Action contract shared by an environment and compatible algorithms."""

    tensor: TensorSpec
    kind: ActionKind = "continuous"
    simplex: bool = False
    cash_index: int | None = None
    mask_key: str | None = "action_mask"
    simplex_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "discrete", "hybrid"):
            raise ValueError(f"Unknown action kind {self.kind!r}.")
        if self.simplex and self.kind == "discrete":
            raise ValueError("A discrete action cannot use the simplex contract.")
        if self.cash_index is not None:
            if len(self.tensor.shape) != 1 or self.tensor.shape[0] is None:
                raise ValueError("cash_index requires a fixed one-dimensional action event shape.")
            if not 0 <= self.cash_index < self.tensor.shape[0]:
                raise ValueError(f"cash_index {self.cash_index} is outside action shape {self.tensor.shape}.")
        if self.simplex_tolerance <= 0:
            raise ValueError("simplex_tolerance must be positive.")

    def validate(self, action: torch.Tensor, *, leading_dims: int = 1, name: str = "action") -> None:
        self.tensor.validate(action, name=name, leading_dims=leading_dims)
        if self.simplex:
            if bool((action < -self.simplex_tolerance).any().item()):
                raise ValueError(f"{name} must be nonnegative for a simplex action.")
            sums = action.sum(dim=-1)
            target = torch.ones_like(sums)
            if not bool(torch.allclose(sums, target, atol=self.simplex_tolerance, rtol=0.0)):
                raise ValueError(f"{name} must sum to one along its final dimension.")
