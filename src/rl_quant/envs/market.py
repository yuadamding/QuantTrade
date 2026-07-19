"""Historical tensor data and the trading-specific observation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch

from rl_quant.rl.types import ObservationBatch


@dataclass(frozen=True)
class HistoricalMarketData:
    """A batch of chronological episodes resident on one torch device.

    ``asset_returns[:, t]`` is the return earned after the action selected from
    ``features[:, t]``.  Features and availability include the final next state,
    hence their time dimension is one longer than the return dimension. Optional
    int64 ``decision_ids`` provide globally unique identities for exact replay
    and cross-seed artifact alignment; callers should encode their composite
    timestamp/universe identity rather than relying on row position.
    """

    features: Mapping[str, torch.Tensor]
    asset_returns: torch.Tensor
    availability: torch.Tensor
    decision_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.asset_returns.ndim != 3:
            raise ValueError("asset_returns must have shape [batch, time, asset].")
        batch_size, horizon, num_assets = self.asset_returns.shape
        if batch_size <= 0 or horizon <= 0 or num_assets <= 0:
            raise ValueError("HistoricalMarketData dimensions must be positive.")
        if not self.asset_returns.is_floating_point():
            raise ValueError("asset_returns must use a floating-point dtype.")
        if not bool(torch.isfinite(self.asset_returns).all().item()):
            raise ValueError("asset_returns must be finite; encode missingness in availability.")
        if bool((self.asset_returns <= -1.0).any().item()):
            raise ValueError("Simple asset returns must be greater than -1.")
        expected_availability = (batch_size, horizon + 1, num_assets)
        if self.availability.shape != expected_availability or self.availability.dtype != torch.bool:
            raise ValueError(
                f"availability must be bool with shape {expected_availability}; got "
                f"{tuple(self.availability.shape)} and {self.availability.dtype}."
            )
        device = self.asset_returns.device
        if self.availability.device != device:
            raise ValueError("asset_returns and availability must share one device.")
        if self.decision_ids is not None:
            if (
                self.decision_ids.shape != (batch_size, horizon)
                or self.decision_ids.dtype != torch.long
                or self.decision_ids.device != device
            ):
                raise ValueError(
                    f"decision_ids must be int64 with shape {(batch_size, horizon)} on {device}."
                )
            if torch.unique(self.decision_ids).numel() != batch_size * horizon:
                raise ValueError(
                    "decision_ids must be globally unique across the batched historical episodes."
                )
        for name, value in self.features.items():
            if not name:
                raise ValueError("Feature names must be non-empty.")
            if value.ndim < 2 or value.shape[:2] != (batch_size, horizon + 1):
                raise ValueError(
                    f"Feature {name!r} needs leading shape {(batch_size, horizon + 1)}; "
                    f"got {tuple(value.shape)}."
                )
            if value.device != device:
                raise ValueError(f"Feature {name!r} is on {value.device}, expected {device}.")
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Feature {name!r} must be finite; provide explicit validity channels.")

    @property
    def batch_size(self) -> int:
        return self.asset_returns.shape[0]

    @property
    def horizon(self) -> int:
        return self.asset_returns.shape[1]

    @property
    def num_assets(self) -> int:
        return self.asset_returns.shape[2]

    @property
    def device(self) -> torch.device:
        return self.asset_returns.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> HistoricalMarketData:
        return HistoricalMarketData(
            features={
                name: value.to(device=device, non_blocking=non_blocking)
                for name, value in self.features.items()
            },
            asset_returns=self.asset_returns.to(device=device, non_blocking=non_blocking),
            availability=self.availability.to(device=device, non_blocking=non_blocking),
            decision_ids=(
                None
                if self.decision_ids is None
                else self.decision_ids.to(device=device, non_blocking=non_blocking)
            ),
        )


class PortfolioObservationAdapter(Protocol):
    """Convert market tensors and portfolio state into an algorithm observation."""

    def build(
        self,
        data: HistoricalMarketData,
        *,
        time_index: int,
        weights: torch.Tensor,
        equity: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> ObservationBatch: ...


@dataclass(frozen=True)
class TensorPortfolioObservationAdapter:
    """Default adapter retaining raw point-in-time fields and portfolio state."""

    weights_key: str = "portfolio_weights"
    equity_key: str = "portfolio_equity"
    time_key: str = "time_index"

    def build(
        self,
        data: HistoricalMarketData,
        *,
        time_index: int,
        weights: torch.Tensor,
        equity: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> ObservationBatch:
        reserved = {self.weights_key, self.equity_key, self.time_key}
        collision = reserved.intersection(data.features)
        if collision:
            raise ValueError(f"Market feature names collide with portfolio state keys: {sorted(collision)}.")
        tensors = {name: value[:, time_index] for name, value in data.features.items()}
        tensors[self.weights_key] = weights
        tensors[self.equity_key] = equity.unsqueeze(-1)
        tensors[self.time_key] = torch.full(
            (data.batch_size, 1), time_index, dtype=torch.long, device=data.device
        )
        return ObservationBatch(
            tensors=tensors,
            action_mask=data.availability[:, time_index],
            episode_start=episode_start,
        )
