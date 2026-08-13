"""Torch-native vectorized historical portfolio environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rl_quant.envs.hold30 import (
    TURNOVER_CAUSES,
    CohortLedger,
    CohortTradeAccounting,
    TurnoverCause,
    reconcile_cash_simplex_roundoff,
    zero_turnover_by_cause,
)
from rl_quant.envs.market import (
    HistoricalMarketData,
    PortfolioObservationAdapter,
    TensorPortfolioObservationAdapter,
)
from rl_quant.execution.portfolio import (
    ImmediateTargetWeightExecution,
    TargetWeightExecutionModel,
    TargetWeightExecutionResult,
    drift_weights,
    one_way_turnover,
)
from rl_quant.rl.specs import ActionSpec, TensorSpec
from rl_quant.rl.types import (
    ActionBatch,
    ObservationBatch,
    RewardComponents,
    TransitionBatch,
)


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool; got {value!r}.")
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be finite and nonnegative; got {value!r}.")
    return out


@dataclass(frozen=True)
class PortfolioConstraints:
    """Convex long-only target-allocation constraints.

    Turnover is one-way turnover: half the L1 distance between two full
    portfolios.  Hard feasibility (for example liquidating an unavailable asset)
    takes precedence when it is mathematically impossible to honor both a hard
    constraint and the turnover cap; that exceptional amount is reported.
    """

    max_asset_weight: float = 1.0
    max_leverage: float = 1.0
    max_turnover: float | None = None
    max_drawdown: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_asset_weight, bool) or isinstance(
            self.max_leverage, bool
        ):
            raise ValueError(
                "max_asset_weight and max_leverage must be numeric, not bool."
            )
        max_weight = float(self.max_asset_weight)
        leverage = float(self.max_leverage)
        if not math.isfinite(max_weight) or not 0 < max_weight <= 1:
            raise ValueError("max_asset_weight must lie in (0, 1].")
        if not math.isfinite(leverage) or not 0 < leverage <= 1:
            raise ValueError("The long-only max_leverage must lie in (0, 1].")
        if self.max_turnover is not None:
            if isinstance(self.max_turnover, bool):
                raise ValueError("max_turnover must be numeric, not bool.")
            turnover = float(self.max_turnover)
            if not math.isfinite(turnover) or not 0 <= turnover <= 1:
                raise ValueError("max_turnover must lie in [0, 1] or be None.")
        if self.max_drawdown is not None:
            if isinstance(self.max_drawdown, bool):
                raise ValueError("max_drawdown must be numeric, not bool.")
            drawdown = float(self.max_drawdown)
            if not math.isfinite(drawdown) or not 0 < drawdown <= 1:
                raise ValueError("max_drawdown must lie in (0, 1] or be None.")
        object.__setattr__(self, "max_asset_weight", max_weight)
        object.__setattr__(self, "max_leverage", leverage)
        object.__setattr__(
            self,
            "max_turnover",
            None if self.max_turnover is None else float(self.max_turnover),
        )
        object.__setattr__(
            self,
            "max_drawdown",
            None if self.max_drawdown is None else float(self.max_drawdown),
        )


@dataclass(frozen=True)
class PortfolioCostModel:
    """Configured return-unit cost assumptions on non-cash traded notional.

    The linear impact term is a modeling sensitivity, not a claim that the
    historical market tensors contain empirical fills or impact observations.
    """

    spread_bps: float = 0.0
    fee_bps: float = 0.0
    impact_bps_per_unit: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spread_bps", _finite_nonnegative("spread_bps", self.spread_bps)
        )
        object.__setattr__(
            self, "fee_bps", _finite_nonnegative("fee_bps", self.fee_bps)
        )
        object.__setattr__(
            self,
            "impact_bps_per_unit",
            _finite_nonnegative("impact_bps_per_unit", self.impact_bps_per_unit),
        )


@dataclass(frozen=True)
class PortfolioEnvState:
    """Complete environment-owned state for exact checkpoint/carry restore."""

    time_index: int
    weights: torch.Tensor
    equity: torch.Tensor
    peak_equity: torch.Tensor
    drawdown: torch.Tensor
    recent_turnover: torch.Tensor
    gross_exposure: torch.Tensor
    risk_halted: torch.Tensor
    done: torch.Tensor
    episode_start: torch.Tensor
    cohort_ledger: CohortLedger
    turnover_by_cause: dict[TurnoverCause, torch.Tensor]
    cumulative_turnover_by_cause: dict[TurnoverCause, torch.Tensor]

    def clone(self) -> PortfolioEnvState:
        return PortfolioEnvState(
            time_index=self.time_index,
            weights=self.weights.clone(),
            equity=self.equity.clone(),
            peak_equity=self.peak_equity.clone(),
            drawdown=self.drawdown.clone(),
            recent_turnover=self.recent_turnover.clone(),
            gross_exposure=self.gross_exposure.clone(),
            risk_halted=self.risk_halted.clone(),
            done=self.done.clone(),
            episode_start=self.episode_start.clone(),
            cohort_ledger=self.cohort_ledger.clone(),
            turnover_by_cause={
                cause: value.clone() for cause, value in self.turnover_by_cause.items()
            },
            cumulative_turnover_by_cause={
                cause: value.clone()
                for cause, value in self.cumulative_turnover_by_cause.items()
            },
        )

    def detach(self) -> PortfolioEnvState:
        return PortfolioEnvState(
            time_index=self.time_index,
            weights=self.weights.detach(),
            equity=self.equity.detach(),
            peak_equity=self.peak_equity.detach(),
            drawdown=self.drawdown.detach(),
            recent_turnover=self.recent_turnover.detach(),
            gross_exposure=self.gross_exposure.detach(),
            risk_halted=self.risk_halted.detach(),
            done=self.done.detach(),
            episode_start=self.episode_start.detach(),
            cohort_ledger=self.cohort_ledger.detach(),
            turnover_by_cause={
                cause: value.detach() for cause, value in self.turnover_by_cause.items()
            },
            cumulative_turnover_by_cause={
                cause: value.detach()
                for cause, value in self.cumulative_turnover_by_cause.items()
            },
        )


@dataclass(frozen=True)
class _ActionProjection:
    target: torch.Tensor
    availability_anchor: torch.Tensor
    risk_anchor: torch.Tensor
    discretionary_target: torch.Tensor
    info: dict[str, torch.Tensor]


class VectorPortfolioEnv:
    """Batched historical environment with one canonical portfolio ledger.

    The action is a requested long-only simplex allocation including CASH.  The
    environment applies availability, per-asset weight, gross-exposure, turnover,
    and drawdown constraints before an immediate target-weight accounting model;
    realizes exactly one chronological return; drifts holdings; and accounts for
    a forced CASH transition at a true terminal or risk halt. Rollout-length
    truncation alone does not force CASH and retains a bootstrap discount.
    """

    def __init__(
        self,
        data: HistoricalMarketData,
        *,
        cash_index: int = 0,
        constraints: PortfolioConstraints | None = None,
        costs: PortfolioCostModel | None = None,
        execution_model: TargetWeightExecutionModel | None = None,
        discount: float = 0.99,
        max_episode_steps: int | None = None,
        observation_adapter: PortfolioObservationAdapter | None = None,
        initial_weights: torch.Tensor | None = None,
        initial_position_age: int = 60,
        track_initial_cohort_units: bool = False,
        terminal_liquidate: bool = True,
    ) -> None:
        if not 0 <= cash_index < data.num_assets:
            raise ValueError(
                f"cash_index {cash_index} is outside {data.num_assets} assets."
            )
        if not math.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("discount must lie in [0, 1].")
        if max_episode_steps is not None and not 0 < max_episode_steps <= data.horizon:
            raise ValueError(f"max_episode_steps must lie in [1, {data.horizon}].")
        if not bool(data.availability[:, :, cash_index].all().item()):
            raise ValueError("CASH must be available at every state.")
        cash_returns = data.asset_returns[:, :, cash_index]
        if not bool(torch.allclose(cash_returns, torch.zeros_like(cash_returns))):
            raise ValueError(
                "The CASH return column must be zero; encode yield as an explicit cash return model."
            )
        if not isinstance(terminal_liquidate, bool):
            raise TypeError("terminal_liquidate must be bool.")
        if not isinstance(track_initial_cohort_units, bool):
            raise TypeError("track_initial_cohort_units must be bool.")
        if (
            isinstance(initial_position_age, bool)
            or not isinstance(initial_position_age, int)
            or not 0 <= initial_position_age <= 60
        ):
            raise ValueError("initial_position_age must be an integer in [0, 60].")

        self.data = data
        self.cash_index = int(cash_index)
        self.constraints = (
            PortfolioConstraints() if constraints is None else constraints
        )
        if costs is not None and execution_model is not None:
            raise ValueError(
                "Pass either costs or execution_model; the execution model is the cost authority."
            )
        self.costs = PortfolioCostModel() if costs is None else costs
        self.execution_model: TargetWeightExecutionModel = (
            ImmediateTargetWeightExecution(
                spread_bps=self.costs.spread_bps,
                fee_bps=self.costs.fee_bps,
                modeled_linear_impact_bps_per_weight=self.costs.impact_bps_per_unit,
            )
            if execution_model is None
            else execution_model
        )
        if not isinstance(self.execution_model, TargetWeightExecutionModel):
            raise TypeError(
                "execution_model must implement TargetWeightExecutionModel.execute()."
            )
        self.discount = float(discount)
        self.max_episode_steps = (
            data.horizon if max_episode_steps is None else int(max_episode_steps)
        )
        self.observation_adapter = (
            TensorPortfolioObservationAdapter()
            if observation_adapter is None
            else observation_adapter
        )
        self.terminal_liquidate = terminal_liquidate
        self.initial_position_age = initial_position_age
        self.track_initial_cohort_units = track_initial_cohort_units
        self.action_spec = ActionSpec(
            tensor=TensorSpec(
                shape=(data.num_assets,),
                dtype=data.asset_returns.dtype,
                low=0.0,
                high=1.0,
            ),
            kind="continuous",
            simplex=True,
            cash_index=cash_index,
        )
        risky = torch.ones(data.num_assets, dtype=torch.bool, device=data.device)
        risky[cash_index] = False
        self._risky_mask = risky
        if initial_weights is None:
            configured_initial = torch.zeros(
                (data.batch_size, data.num_assets),
                dtype=data.asset_returns.dtype,
                device=data.device,
            )
            configured_initial[:, cash_index] = 1.0
        else:
            configured_initial = torch.as_tensor(
                initial_weights,
                dtype=data.asset_returns.dtype,
                device=data.device,
            )
            if configured_initial.ndim == 1:
                configured_initial = (
                    configured_initial.unsqueeze(0).expand(data.batch_size, -1).clone()
                )
            self.action_spec.validate(configured_initial, name="initial_weights")
            if bool(
                (configured_initial.masked_select(~data.availability[:, 0]) > 0)
                .any()
                .item()
            ):
                raise ValueError(
                    "initial_weights allocate to an asset unavailable at the initial state."
                )
        self._initial_weights = configured_initial.clone()
        self._time_index = 0
        self._weights = self._initial_weights.clone()
        self._equity = torch.ones(
            data.batch_size, dtype=data.asset_returns.dtype, device=data.device
        )
        self._peak_equity = torch.ones_like(self._equity)
        self._drawdown = torch.zeros_like(self._equity)
        self._recent_turnover = torch.zeros_like(self._equity)
        self._gross_exposure = self._weights[:, self._risky_mask].sum(dim=-1)
        self._risk_halted = torch.zeros(
            data.batch_size, dtype=torch.bool, device=data.device
        )
        self._done = torch.ones(data.batch_size, dtype=torch.bool, device=data.device)
        self._age_ledger = CohortLedger.from_weights(
            self._weights,
            cash_index=self.cash_index,
            initial_age=self.initial_position_age,
            track_initial_units=self.track_initial_cohort_units,
        )
        self._turnover_by_cause = zero_turnover_by_cause(self._equity)
        self._cumulative_turnover_by_cause = zero_turnover_by_cause(self._equity)
        self._observation: ObservationBatch | None = None

    @property
    def batch_size(self) -> int:
        return self.data.batch_size

    @property
    def time_index(self) -> int:
        return self._time_index

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def equity(self) -> torch.Tensor:
        return self._equity

    @property
    def peak_equity(self) -> torch.Tensor:
        return self._peak_equity

    @property
    def drawdown(self) -> torch.Tensor:
        return self._drawdown

    @property
    def recent_turnover(self) -> torch.Tensor:
        return self._recent_turnover

    @property
    def gross_exposure(self) -> torch.Tensor:
        return self._gross_exposure

    @property
    def risk_halted(self) -> torch.Tensor:
        return self._risk_halted

    @property
    def age_ledger(self) -> CohortLedger:
        return self._age_ledger

    @property
    def turnover_by_cause(self) -> dict[TurnoverCause, torch.Tensor]:
        return {
            cause: value.clone() for cause, value in self._turnover_by_cause.items()
        }

    @property
    def cumulative_turnover_by_cause(self) -> dict[TurnoverCause, torch.Tensor]:
        return {
            cause: value.clone()
            for cause, value in self._cumulative_turnover_by_cause.items()
        }

    def _build_observation(
        self, *, time_index: int, episode_start: torch.Tensor
    ) -> ObservationBatch:
        """Add environment-owned risk state without changing adapter signatures."""

        adapted = self.observation_adapter.build(
            self.data,
            time_index=time_index,
            weights=self._weights,
            equity=self._equity,
            episode_start=episode_start,
        )
        available = self.data.availability[:, time_index]
        cash_only = torch.zeros_like(available)
        cash_only[:, self.cash_index] = True
        hard_action_mask = torch.where(
            self._risk_halted.unsqueeze(-1), cash_only, available
        )
        if adapted.action_mask is not None:
            if adapted.action_mask.shape != hard_action_mask.shape:
                raise ValueError(
                    "Portfolio observation adapter action_mask must match [batch, asset] availability."
                )
            hard_action_mask &= adapted.action_mask
            # CASH is the environment's hard fallback and can never be masked by
            # a presentation adapter.
            hard_action_mask[:, self.cash_index] = True

        dtype = self._equity.dtype
        device = self._equity.device

        def _constant(value: float) -> torch.Tensor:
            return torch.full((self.batch_size, 1), value, dtype=dtype, device=device)

        max_turnover = (
            1.0
            if self.constraints.max_turnover is None
            else self.constraints.max_turnover
        )
        max_drawdown = (
            1.0
            if self.constraints.max_drawdown is None
            else self.constraints.max_drawdown
        )
        state = {
            "portfolio_peak_equity": self._peak_equity.unsqueeze(-1),
            "portfolio_drawdown": self._drawdown.unsqueeze(-1),
            "portfolio_recent_turnover": self._recent_turnover.unsqueeze(-1),
            "portfolio_gross_exposure": self._gross_exposure.unsqueeze(-1),
            "portfolio_cash_weight": self._weights[:, self.cash_index].unsqueeze(-1),
            "portfolio_position_age_summary": self._age_ledger.age_summaries(),
            "constraint_max_asset_weight": _constant(self.constraints.max_asset_weight),
            "constraint_max_gross_exposure": _constant(self.constraints.max_leverage),
            "constraint_max_turnover": _constant(max_turnover),
            "constraint_turnover_enabled": torch.full(
                (self.batch_size, 1),
                self.constraints.max_turnover is not None,
                dtype=torch.bool,
                device=device,
            ),
            "constraint_max_drawdown": _constant(max_drawdown),
            "constraint_drawdown_enabled": torch.full(
                (self.batch_size, 1),
                self.constraints.max_drawdown is not None,
                dtype=torch.bool,
                device=device,
            ),
            "constraint_risk_halted": self._risk_halted.unsqueeze(-1),
            "constraint_valid_action_fraction": hard_action_mask.to(dtype=dtype).mean(
                dim=-1,
                keepdim=True,
            ),
        }
        collision = set(adapted.tensors).intersection(state)
        if collision:
            raise ValueError(
                f"Observation adapter tensors collide with environment-owned risk state: {sorted(collision)}."
            )
        return ObservationBatch(
            tensors={**adapted.tensors, **state},
            action_mask=hard_action_mask,
            episode_start=adapted.episode_start,
        )

    @torch.no_grad()
    def reset(self) -> tuple[ObservationBatch, dict[str, torch.Tensor]]:
        self._time_index = 0
        self._weights = self._initial_weights.clone()
        self._equity = torch.ones_like(self._equity)
        self._peak_equity = torch.ones_like(self._equity)
        self._drawdown = torch.zeros_like(self._equity)
        self._recent_turnover = torch.zeros_like(self._equity)
        self._gross_exposure = self._weights[:, self._risky_mask].sum(dim=-1)
        self._risk_halted = torch.zeros_like(self._risk_halted)
        self._done = torch.zeros_like(self._done)
        self._age_ledger = CohortLedger.from_weights(
            self._weights,
            cash_index=self.cash_index,
            initial_age=self.initial_position_age,
            track_initial_units=self.track_initial_cohort_units,
        )
        self._turnover_by_cause = zero_turnover_by_cause(self._equity)
        self._cumulative_turnover_by_cause = zero_turnover_by_cause(self._equity)
        starts = torch.ones(self.batch_size, dtype=torch.bool, device=self.data.device)
        self._observation = self._build_observation(time_index=0, episode_start=starts)
        return self._observation, {
            "equity": self._equity.clone(),
            "peak_equity": self._peak_equity.clone(),
            "drawdown": self._drawdown.clone(),
            "weights": self._weights.clone(),
            "risk_halted": self._risk_halted.clone(),
            "cohort_economic_value": self._age_ledger.economic_value.clone(),
            "cohort_retention_units": self._age_ledger.retention_units.clone(),
        }

    def capture_state(self, *, detach: bool = True) -> PortfolioEnvState:
        """Capture every environment-owned field without an economic reset.

        ``detach=True`` is the intended truncated-backpropagation boundary:
        numeric portfolio and cohort state carries unchanged while its prior
        autograd history is discarded.
        """

        if self._observation is None:
            raise RuntimeError("Call reset() before capture_state().")
        episode_start = (
            self._observation.episode_start
            if self._observation.episode_start is not None
            else torch.zeros(self.batch_size, dtype=torch.bool, device=self.data.device)
        )
        state = PortfolioEnvState(
            time_index=self._time_index,
            weights=self._weights,
            equity=self._equity,
            peak_equity=self._peak_equity,
            drawdown=self._drawdown,
            recent_turnover=self._recent_turnover,
            gross_exposure=self._gross_exposure,
            risk_halted=self._risk_halted,
            done=self._done,
            episode_start=episode_start,
            cohort_ledger=self._age_ledger,
            turnover_by_cause=self._turnover_by_cause,
            cumulative_turnover_by_cause=self._cumulative_turnover_by_cause,
        )
        return state.detach().clone() if detach else state.clone()

    @torch.no_grad()
    def restore_state(
        self,
        state: PortfolioEnvState,
    ) -> tuple[ObservationBatch, dict[str, torch.Tensor]]:
        """Restore a checkpoint/carry state exactly, without re-endowment."""

        if not isinstance(state, PortfolioEnvState):
            raise TypeError("state must be PortfolioEnvState.")
        if (
            isinstance(state.time_index, bool)
            or not isinstance(state.time_index, int)
            or not 0 <= state.time_index <= self.data.horizon
        ):
            raise ValueError("state.time_index is outside this market trajectory.")
        expected_weights = (self.batch_size, self.data.num_assets)
        expected_batch = (self.batch_size,)
        tensor_shapes = {
            "weights": (state.weights, expected_weights),
            "equity": (state.equity, expected_batch),
            "peak_equity": (state.peak_equity, expected_batch),
            "drawdown": (state.drawdown, expected_batch),
            "recent_turnover": (state.recent_turnover, expected_batch),
            "gross_exposure": (state.gross_exposure, expected_batch),
            "risk_halted": (state.risk_halted, expected_batch),
            "done": (state.done, expected_batch),
            "episode_start": (state.episode_start, expected_batch),
        }
        for name, (value, expected_shape) in tensor_shapes.items():
            if value.shape != expected_shape:
                raise ValueError(f"state.{name} must have shape {expected_shape}.")
            if value.device != self.data.device:
                raise ValueError(f"state.{name} must be on {self.data.device}.")
        if state.weights.dtype != self.data.asset_returns.dtype:
            raise ValueError("state.weights must share the market return dtype.")
        if (
            state.cohort_ledger.cash_index != self.cash_index
            or state.cohort_ledger.economic_value.shape
            != (self.batch_size, self.data.num_assets, 61)
            or state.cohort_ledger.economic_value.dtype != self.data.asset_returns.dtype
            or state.cohort_ledger.economic_value.device != self.data.device
        ):
            raise ValueError(
                "state cohort ledger is incompatible with this environment."
            )
        if set(state.turnover_by_cause) != set(TURNOVER_CAUSES) or set(
            state.cumulative_turnover_by_cause
        ) != set(TURNOVER_CAUSES):
            raise ValueError(
                "state turnover mappings must contain every TurnoverCause exactly once."
            )
        for mapping in (state.turnover_by_cause, state.cumulative_turnover_by_cause):
            if any(
                value.shape != expected_batch
                or value.dtype != self.data.asset_returns.dtype
                or value.device != self.data.device
                for value in mapping.values()
            ):
                raise ValueError(
                    "state turnover mappings must match the environment batch, dtype, and device."
                )
        state.cohort_ledger.assert_reconciles(state.weights)

        copied = state.detach().clone()
        self._time_index = copied.time_index
        self._weights = copied.weights
        self._equity = copied.equity
        self._peak_equity = copied.peak_equity
        self._drawdown = copied.drawdown
        self._recent_turnover = copied.recent_turnover
        self._gross_exposure = copied.gross_exposure
        self._risk_halted = copied.risk_halted
        self._done = copied.done
        self._age_ledger = copied.cohort_ledger
        self._turnover_by_cause = copied.turnover_by_cause
        self._cumulative_turnover_by_cause = copied.cumulative_turnover_by_cause
        self._observation = self._build_observation(
            time_index=self._time_index,
            episode_start=copied.episode_start,
        )
        return self._observation, {
            "equity": self._equity.clone(),
            "weights": self._weights.clone(),
            "cohort_economic_value": self._age_ledger.economic_value.clone(),
            "cohort_retention_units": self._age_ledger.retention_units.clone(),
        }

    def _hard_project(
        self, weights: torch.Tensor, availability: torch.Tensor
    ) -> torch.Tensor:
        masked = torch.where(
            availability, weights.clamp_min(0.0), torch.zeros_like(weights)
        )
        risky = masked[:, self._risky_mask].clamp_max(self.constraints.max_asset_weight)
        exposure = risky.sum(dim=-1, keepdim=True)
        leverage = torch.as_tensor(
            self.constraints.max_leverage, dtype=weights.dtype, device=weights.device
        )
        scale = torch.where(
            exposure > leverage,
            leverage / exposure.clamp_min(1e-12),
            torch.ones_like(exposure),
        )
        risky = risky * scale
        projected = torch.zeros_like(weights)
        projected[:, self._risky_mask] = risky
        projected[:, self.cash_index] = 1.0 - risky.sum(dim=-1)
        return reconcile_cash_simplex_roundoff(
            projected,
            cash_index=self.cash_index,
            risky_gross_limit=torch.full(
                (weights.shape[0],),
                self.constraints.max_leverage,
                dtype=weights.dtype,
                device=weights.device,
            ),
        )

    def _availability_project(
        self, weights: torch.Tensor, availability: torch.Tensor
    ) -> torch.Tensor:
        """Liquidate only unavailable risky assets and leave other weights unchanged."""

        projected = weights.clone()
        projected[:, self._risky_mask] = torch.where(
            availability[:, self._risky_mask],
            projected[:, self._risky_mask],
            torch.zeros_like(projected[:, self._risky_mask]),
        )
        projected[:, self.cash_index] = 1.0 - projected[:, self._risky_mask].sum(dim=-1)
        return reconcile_cash_simplex_roundoff(
            projected,
            cash_index=self.cash_index,
            risky_gross_limit=torch.full(
                (weights.shape[0],),
                self.constraints.max_leverage,
                dtype=weights.dtype,
                device=weights.device,
            ),
        )

    @staticmethod
    def _one_way_turnover(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        return one_way_turnover(new, old)

    def _project_action_stages(
        self,
        requested: torch.Tensor,
        availability: torch.Tensor,
    ) -> _ActionProjection:
        candidate = self._hard_project(requested, availability)
        # The feasible anchor handles non-negotiable availability/weight/leverage
        # violations caused by market drift before applying discretionary turnover.
        availability_anchor = self._availability_project(self._weights, availability)
        anchor = self._hard_project(self._weights, availability)
        forced = self._one_way_turnover(anchor, self._weights)
        discretionary = self._one_way_turnover(candidate, anchor)
        if self.constraints.max_turnover is None:
            alpha = torch.ones_like(discretionary)
            forced_excess = torch.zeros_like(forced)
        else:
            cap = torch.full_like(forced, self.constraints.max_turnover)
            remaining = (cap - forced).clamp_min(0.0)
            alpha = torch.where(
                discretionary > 0,
                (remaining / discretionary.clamp_min(1e-12)).clamp(max=1.0),
                torch.ones_like(discretionary),
            )
            forced_excess = (forced - cap).clamp_min(0.0)
        projected = reconcile_cash_simplex_roundoff(
            anchor + alpha.unsqueeze(-1) * (candidate - anchor),
            cash_index=self.cash_index,
            risky_gross_limit=torch.full(
                (requested.shape[0],),
                self.constraints.max_leverage,
                dtype=requested.dtype,
                device=requested.device,
            ),
        )
        fallback = torch.zeros_like(projected)
        fallback[:, self.cash_index] = 1.0
        risk_override_turnover = self._one_way_turnover(fallback, projected)
        target = torch.where(self._risk_halted.unsqueeze(-1), fallback, projected)
        # Once risk is already halted, suppress hypothetical intermediate
        # trades.  The only economic transition is the risk-forced CASH target.
        active = ~self._risk_halted.unsqueeze(-1)
        availability_stage = torch.where(active, availability_anchor, self._weights)
        risk_stage = torch.where(active, anchor, self._weights)
        discretionary_stage = torch.where(active, projected, self._weights)
        return _ActionProjection(
            target=target,
            availability_anchor=availability_stage,
            risk_anchor=risk_stage,
            discretionary_target=discretionary_stage,
            info={
                "forced_turnover": forced,
                "forced_turnover_excess": forced_excess,
                "requested_projection_distance": self._one_way_turnover(
                    target, requested
                ),
                "risk_override": self._risk_halted.clone(),
                "risk_override_turnover": torch.where(
                    self._risk_halted,
                    risk_override_turnover,
                    torch.zeros_like(risk_override_turnover),
                ),
            },
        )

    def _project_action(
        self,
        requested: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projection = self._project_action_stages(requested, availability)
        return projection.target, projection.info

    @staticmethod
    def _record_trade(
        ledger: CohortLedger,
        target: torch.Tensor,
        cause: TurnoverCause,
        turnover_by_cause: dict[TurnoverCause, torch.Tensor],
    ) -> tuple[CohortLedger, CohortTradeAccounting]:
        next_ledger, accounting = ledger.trade_to(target, cause=cause)
        turnover_by_cause[cause] = turnover_by_cause[cause] + accounting.turnover
        return next_ledger, accounting

    def _execute_target(
        self,
        current_weights: torch.Tensor,
        target_weights: torch.Tensor,
    ) -> TargetWeightExecutionResult:
        """Run the authoritative cost model for an environment-approved target."""

        result = self.execution_model.execute(
            current_weights,
            target_weights,
            cash_index=self.cash_index,
        )
        if not isinstance(result, TargetWeightExecutionResult):
            raise TypeError(
                "execution_model.execute() must return TargetWeightExecutionResult."
            )
        expected = (self.batch_size,)
        for name, value in (
            ("execution_cost", result.execution_cost),
            ("modeled_impact_cost", result.modeled_impact_cost),
            ("traded_notional", result.traded_notional),
        ):
            if (
                value.shape != expected
                or value.dtype != target_weights.dtype
                or value.device != target_weights.device
            ):
                raise ValueError(
                    f"Execution result {name} must have shape {expected}, dtype {target_weights.dtype}, "
                    f"on {target_weights.device}."
                )
        unchanged = torch.eq(current_weights, target_weights).all(dim=-1)
        charged_unchanged = (
            result.execution_cost + result.modeled_impact_cost + result.traded_notional
        )[unchanged]
        if bool((charged_unchanged != 0).any().item()):
            raise ValueError(
                "Execution models must report zero trade and cost when target weights are unchanged."
            )
        return result

    @torch.no_grad()
    def step(self, action: ActionBatch | torch.Tensor) -> TransitionBatch:
        if self._observation is None:
            raise RuntimeError("Call reset() before step().")
        if bool(self._done.any().item()):
            raise RuntimeError(
                "The episode is done; call reset() before another step()."
            )
        action_batch = (
            action if isinstance(action, ActionBatch) else ActionBatch(action=action)
        )
        if (
            action_batch.batch_size != self.batch_size
            or action_batch.device != self.data.device
        ):
            raise ValueError(
                f"Action must have batch size {self.batch_size} on {self.data.device}."
            )
        self.action_spec.validate(action_batch.action)
        # Historical RL interaction is intentionally non-differentiable. Actor
        # gradients flow through stored log probabilities/Q estimates, not
        # backward through a market simulator across an entire rollout.
        action_batch = action_batch.detach()

        old_observation = self._observation
        old_equity = self._equity
        old_peak_equity = self._peak_equity
        old_weights = self._weights
        availability = self.data.availability[:, self._time_index]
        projection = self._project_action_stages(action_batch.action, availability)
        target_weights = projection.target
        projection_info = projection.info
        cause_turnover = zero_turnover_by_cause(old_equity)
        ledger = self._age_ledger
        ledger, _availability_accounting = self._record_trade(
            ledger,
            projection.availability_anchor,
            TurnoverCause.AVAILABILITY_FORCED,
            cause_turnover,
        )
        ledger, _risk_anchor_accounting = self._record_trade(
            ledger,
            projection.risk_anchor,
            TurnoverCause.RISK_FORCED,
            cause_turnover,
        )
        ledger, discretionary_accounting = self._record_trade(
            ledger,
            projection.discretionary_target,
            TurnoverCause.DISCRETIONARY,
            cause_turnover,
        )
        ledger, _risk_override_accounting = self._record_trade(
            ledger,
            target_weights,
            TurnoverCause.RISK_FORCED,
            cause_turnover,
        )
        ledger.assert_reconciles(target_weights)
        # Risk and feasibility are resolved before the execution model sees the
        # target, so neither a policy nor an execution-cost implementation can
        # substitute an unsafe allocation.
        self.action_spec.validate(target_weights, name="environment-approved target")
        execution = self._execute_target(old_weights, target_weights)

        period_returns = self.data.asset_returns[:, self._time_index]
        gross_return = (target_weights * period_returns).sum(dim=-1)
        growth = 1.0 + gross_return
        if bool((growth <= 0).any().item()):
            raise RuntimeError(
                "Portfolio gross return reached -100%; weights cannot be drifted."
            )
        drifted = reconcile_cash_simplex_roundoff(
            drift_weights(target_weights, period_returns),
            cash_index=self.cash_index,
        )
        ledger = ledger.age_and_drift(period_returns)
        ledger.assert_reconciles(drifted)

        base_net_return = (
            gross_return - execution.execution_cost - execution.modeled_impact_cost
        )
        if bool((base_net_return <= -1.0).any().item()):
            raise RuntimeError(
                "Net return reached -100% before forced fallback; check execution costs."
            )
        provisional_equity = old_equity * (1.0 + base_net_return)
        provisional_peak = torch.maximum(old_peak_equity, provisional_equity)
        provisional_drawdown = (
            1.0 - provisional_equity / provisional_peak.clamp_min(1e-12)
        ).clamp(
            min=0.0,
            max=1.0,
        )
        if self.constraints.max_drawdown is None:
            drawdown_breach = torch.zeros_like(self._risk_halted)
        else:
            drawdown_breach = provisional_drawdown >= self.constraints.max_drawdown
        risk_halt_triggered = drawdown_breach & ~self._risk_halted
        next_risk_halted = self._risk_halted | drawdown_breach

        next_index = self._time_index + 1
        terminated = torch.full(
            (self.batch_size,),
            next_index >= self.data.horizon,
            dtype=torch.bool,
            device=self.data.device,
        )
        truncated = torch.full(
            (self.batch_size,),
            next_index >= self.max_episode_steps and next_index < self.data.horizon,
            dtype=torch.bool,
            device=self.data.device,
        )
        terminal_required = terminated & self.terminal_liquidate
        risk_required = risk_halt_triggered & ~terminal_required
        fallback_required = terminal_required | risk_required
        cash_target = torch.zeros_like(drifted)
        cash_target[:, self.cash_index] = 1.0
        terminal_target = torch.where(
            terminal_required.unsqueeze(-1), cash_target, drifted
        )
        ledger, _terminal_accounting = self._record_trade(
            ledger,
            terminal_target,
            TurnoverCause.TERMINAL,
            cause_turnover,
        )
        next_weights = torch.where(
            risk_required.unsqueeze(-1), cash_target, terminal_target
        )
        ledger, _risk_fallback_accounting = self._record_trade(
            ledger,
            next_weights,
            TurnoverCause.RISK_FORCED,
            cause_turnover,
        )
        ledger.assert_reconciles(next_weights)
        if bool(fallback_required.any().item()):
            fallback_execution = self._execute_target(drifted, next_weights)
        else:
            zero = torch.zeros_like(gross_return)
            fallback_execution = TargetWeightExecutionResult(
                execution_cost=zero,
                modeled_impact_cost=zero,
                traded_notional=zero,
            )
        fallback_cost_fraction = (
            fallback_execution.execution_cost + fallback_execution.modeled_impact_cost
        )
        # Fallback executes after the period return. Execution models express
        # cost as a fraction of the then-current portfolio, while RewardComponents
        # is measured relative to equity at the beginning of the transition.
        # Scale once so the additive ledger is exactly equivalent to sequential
        # ``provisional_equity * (1 - fallback_cost_fraction)`` accounting.
        liquidation_cost = fallback_cost_fraction * (1.0 + base_net_return)

        zeros = torch.zeros_like(gross_return)
        rewards = RewardComponents(
            gross_return=gross_return,
            execution_cost=execution.execution_cost,
            impact_cost=execution.modeled_impact_cost,
            risk_penalty=zeros,
            constraint_penalty=zeros,
            liquidation_cost=liquidation_cost,
        )
        if bool((rewards.total <= -1.0).any().item()):
            raise RuntimeError(
                "Net one-step reward reached -100%; check return data and execution costs."
            )
        self._equity = old_equity * (1.0 + rewards.total)
        self._peak_equity = torch.maximum(old_peak_equity, self._equity)
        self._drawdown = (
            1.0 - self._equity / self._peak_equity.clamp_min(1e-12)
        ).clamp(
            min=0.0,
            max=1.0,
        )
        self._weights = next_weights
        self._age_ledger = ledger
        base_turnover = self._one_way_turnover(target_weights, old_weights)
        fallback_turnover = self._one_way_turnover(next_weights, drifted)
        self._recent_turnover = base_turnover + fallback_turnover
        cause_total = torch.stack(tuple(cause_turnover.values()), dim=0).sum(dim=0)
        if not bool(
            torch.allclose(cause_total, self._recent_turnover, atol=1e-6, rtol=1e-6)
        ):
            error = (cause_total - self._recent_turnover).abs().max().item()
            raise RuntimeError(
                f"Cause-typed turnover failed to reconcile (max diff {error:g})."
            )
        self._turnover_by_cause = cause_turnover
        self._cumulative_turnover_by_cause = {
            cause: self._cumulative_turnover_by_cause[cause] + cause_turnover[cause]
            for cause in TURNOVER_CAUSES
        }
        self._gross_exposure = self._weights[:, self._risky_mask].sum(dim=-1)
        self._risk_halted = next_risk_halted
        self._time_index = next_index
        self._done = terminated | truncated
        next_starts = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.data.device
        )
        self._observation = self._build_observation(
            time_index=next_index, episode_start=next_starts
        )
        discount = torch.full_like(gross_return, self.discount)
        discount = torch.where(terminated, torch.zeros_like(discount), discount)
        info = {
            **projection_info,
            "equity_before": old_equity,
            "equity_after": self._equity,
            "peak_equity": self._peak_equity,
            "drawdown": self._drawdown,
            "one_way_turnover": base_turnover,
            "recent_turnover": self._recent_turnover,
            "traded_notional": execution.traded_notional,
            "liquidation_notional": fallback_execution.traded_notional,
            "liquidation_execution_cost": fallback_execution.execution_cost,
            "liquidation_modeled_impact_cost": fallback_execution.modeled_impact_cost,
            "liquidation_cost_return_units": liquidation_cost,
            "gross_exposure": target_weights[:, self._risky_mask].sum(dim=-1),
            "next_gross_exposure": self._gross_exposure,
            "risk_halt_triggered": risk_halt_triggered,
            "risk_halted": self._risk_halted,
            "early_exit_notional": discretionary_accounting.early_exit_notional,
            "early_exit_units": discretionary_accounting.early_exit_units,
            "turnover_reconciliation_error": cause_total - self._recent_turnover,
            "position_age_summary": self._age_ledger.age_summaries(),
            "environment_index": torch.arange(
                self.batch_size, dtype=torch.long, device=self.data.device
            ),
            "decision_index": torch.full(
                (self.batch_size,),
                next_index - 1,
                dtype=torch.long,
                device=self.data.device,
            ),
            "step_index": torch.full(
                (self.batch_size,),
                self._time_index,
                dtype=torch.long,
                device=self.data.device,
            ),
            **{
                f"execution_{name}": value
                for name, value in execution.diagnostics.items()
            },
            **{
                f"liquidation_execution_{name}": value
                for name, value in fallback_execution.diagnostics.items()
            },
            **{
                f"turnover_{cause.value}": cause_turnover[cause]
                for cause in TURNOVER_CAUSES
            },
            **{
                f"cumulative_turnover_{cause.value}": self._cumulative_turnover_by_cause[
                    cause
                ]
                for cause in TURNOVER_CAUSES
            },
        }
        if self.data.decision_ids is not None:
            info["decision_id"] = self.data.decision_ids[:, next_index - 1]
        return TransitionBatch(
            observation=old_observation,
            action=action_batch,
            executed_action=target_weights,
            rewards=rewards,
            next_observation=self._observation,
            terminated=terminated,
            truncated=truncated,
            discount=discount,
            info=info,
        )
