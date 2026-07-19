"""Torch-native vectorized historical portfolio environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

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
from rl_quant.rl.types import ActionBatch, ObservationBatch, RewardComponents, TransitionBatch


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
        if isinstance(self.max_asset_weight, bool) or isinstance(self.max_leverage, bool):
            raise ValueError("max_asset_weight and max_leverage must be numeric, not bool.")
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
        object.__setattr__(self, "max_turnover", None if self.max_turnover is None else float(self.max_turnover))
        object.__setattr__(self, "max_drawdown", None if self.max_drawdown is None else float(self.max_drawdown))


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
        object.__setattr__(self, "spread_bps", _finite_nonnegative("spread_bps", self.spread_bps))
        object.__setattr__(self, "fee_bps", _finite_nonnegative("fee_bps", self.fee_bps))
        object.__setattr__(
            self,
            "impact_bps_per_unit",
            _finite_nonnegative("impact_bps_per_unit", self.impact_bps_per_unit),
        )


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
    ) -> None:
        if not 0 <= cash_index < data.num_assets:
            raise ValueError(f"cash_index {cash_index} is outside {data.num_assets} assets.")
        if not math.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("discount must lie in [0, 1].")
        if max_episode_steps is not None and not 0 < max_episode_steps <= data.horizon:
            raise ValueError(f"max_episode_steps must lie in [1, {data.horizon}].")
        if not bool(data.availability[:, :, cash_index].all().item()):
            raise ValueError("CASH must be available at every state.")
        cash_returns = data.asset_returns[:, :, cash_index]
        if not bool(torch.allclose(cash_returns, torch.zeros_like(cash_returns))):
            raise ValueError("The CASH return column must be zero; encode yield as an explicit cash return model.")

        self.data = data
        self.cash_index = int(cash_index)
        self.constraints = PortfolioConstraints() if constraints is None else constraints
        if costs is not None and execution_model is not None:
            raise ValueError("Pass either costs or execution_model; the execution model is the cost authority.")
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
            raise TypeError("execution_model must implement TargetWeightExecutionModel.execute().")
        self.discount = float(discount)
        self.max_episode_steps = data.horizon if max_episode_steps is None else int(max_episode_steps)
        self.observation_adapter = (
            TensorPortfolioObservationAdapter() if observation_adapter is None else observation_adapter
        )
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
        self._time_index = 0
        self._weights = torch.zeros(
            (data.batch_size, data.num_assets), dtype=data.asset_returns.dtype, device=data.device
        )
        self._equity = torch.ones(data.batch_size, dtype=data.asset_returns.dtype, device=data.device)
        self._peak_equity = torch.ones_like(self._equity)
        self._drawdown = torch.zeros_like(self._equity)
        self._recent_turnover = torch.zeros_like(self._equity)
        self._gross_exposure = torch.zeros_like(self._equity)
        self._risk_halted = torch.zeros(data.batch_size, dtype=torch.bool, device=data.device)
        self._done = torch.ones(data.batch_size, dtype=torch.bool, device=data.device)
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

    def _build_observation(self, *, time_index: int, episode_start: torch.Tensor) -> ObservationBatch:
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
        hard_action_mask = torch.where(self._risk_halted.unsqueeze(-1), cash_only, available)
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

        max_turnover = 1.0 if self.constraints.max_turnover is None else self.constraints.max_turnover
        max_drawdown = 1.0 if self.constraints.max_drawdown is None else self.constraints.max_drawdown
        state = {
            "portfolio_peak_equity": self._peak_equity.unsqueeze(-1),
            "portfolio_drawdown": self._drawdown.unsqueeze(-1),
            "portfolio_recent_turnover": self._recent_turnover.unsqueeze(-1),
            "portfolio_gross_exposure": self._gross_exposure.unsqueeze(-1),
            "portfolio_cash_weight": self._weights[:, self.cash_index].unsqueeze(-1),
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
        self._weights = torch.zeros_like(self._weights)
        self._weights[:, self.cash_index] = 1.0
        self._equity = torch.ones_like(self._equity)
        self._peak_equity = torch.ones_like(self._equity)
        self._drawdown = torch.zeros_like(self._equity)
        self._recent_turnover = torch.zeros_like(self._equity)
        self._gross_exposure = torch.zeros_like(self._equity)
        self._risk_halted = torch.zeros_like(self._risk_halted)
        self._done = torch.zeros_like(self._done)
        starts = torch.ones(self.batch_size, dtype=torch.bool, device=self.data.device)
        self._observation = self._build_observation(time_index=0, episode_start=starts)
        return self._observation, {
            "equity": self._equity.clone(),
            "peak_equity": self._peak_equity.clone(),
            "drawdown": self._drawdown.clone(),
            "weights": self._weights.clone(),
            "risk_halted": self._risk_halted.clone(),
        }

    def _hard_project(self, weights: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        masked = torch.where(availability, weights.clamp_min(0.0), torch.zeros_like(weights))
        risky = masked[:, self._risky_mask].clamp_max(self.constraints.max_asset_weight)
        exposure = risky.sum(dim=-1, keepdim=True)
        leverage = torch.as_tensor(
            self.constraints.max_leverage, dtype=weights.dtype, device=weights.device
        )
        scale = torch.where(exposure > leverage, leverage / exposure.clamp_min(1e-12), torch.ones_like(exposure))
        risky = risky * scale
        projected = torch.zeros_like(weights)
        projected[:, self._risky_mask] = risky
        projected[:, self.cash_index] = 1.0 - risky.sum(dim=-1)
        return projected

    @staticmethod
    def _one_way_turnover(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        return one_way_turnover(new, old)

    def _project_action(
        self, requested: torch.Tensor, availability: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        candidate = self._hard_project(requested, availability)
        # The feasible anchor handles non-negotiable availability/weight/leverage
        # violations caused by market drift before applying discretionary turnover.
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
        projected = anchor + alpha.unsqueeze(-1) * (candidate - anchor)
        fallback = torch.zeros_like(projected)
        fallback[:, self.cash_index] = 1.0
        risk_override_turnover = self._one_way_turnover(fallback, projected)
        target = torch.where(self._risk_halted.unsqueeze(-1), fallback, projected)
        return target, {
            "forced_turnover": forced,
            "forced_turnover_excess": forced_excess,
            "requested_projection_distance": self._one_way_turnover(target, requested),
            "risk_override": self._risk_halted.clone(),
            "risk_override_turnover": torch.where(
                self._risk_halted,
                risk_override_turnover,
                torch.zeros_like(risk_override_turnover),
            ),
        }

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
            raise TypeError("execution_model.execute() must return TargetWeightExecutionResult.")
        expected = (self.batch_size,)
        for name, value in (
            ("execution_cost", result.execution_cost),
            ("modeled_impact_cost", result.modeled_impact_cost),
            ("traded_notional", result.traded_notional),
        ):
            if value.shape != expected or value.dtype != target_weights.dtype or value.device != target_weights.device:
                raise ValueError(
                    f"Execution result {name} must have shape {expected}, dtype {target_weights.dtype}, "
                    f"on {target_weights.device}."
                )
        unchanged = torch.eq(current_weights, target_weights).all(dim=-1)
        charged_unchanged = (
            result.execution_cost + result.modeled_impact_cost + result.traded_notional
        )[unchanged]
        if bool((charged_unchanged != 0).any().item()):
            raise ValueError("Execution models must report zero trade and cost when target weights are unchanged.")
        return result

    @torch.no_grad()
    def step(self, action: ActionBatch | torch.Tensor) -> TransitionBatch:
        if self._observation is None:
            raise RuntimeError("Call reset() before step().")
        if bool(self._done.any().item()):
            raise RuntimeError("The episode is done; call reset() before another step().")
        action_batch = action if isinstance(action, ActionBatch) else ActionBatch(action=action)
        if action_batch.batch_size != self.batch_size or action_batch.device != self.data.device:
            raise ValueError(f"Action must have batch size {self.batch_size} on {self.data.device}.")
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
        target_weights, projection_info = self._project_action(action_batch.action, availability)
        # Risk and feasibility are resolved before the execution model sees the
        # target, so neither a policy nor an execution-cost implementation can
        # substitute an unsafe allocation.
        self.action_spec.validate(target_weights, name="environment-approved target")
        execution = self._execute_target(old_weights, target_weights)

        period_returns = self.data.asset_returns[:, self._time_index]
        gross_return = (target_weights * period_returns).sum(dim=-1)
        growth = 1.0 + gross_return
        if bool((growth <= 0).any().item()):
            raise RuntimeError("Portfolio gross return reached -100%; weights cannot be drifted.")
        drifted = drift_weights(target_weights, period_returns)

        base_net_return = gross_return - execution.execution_cost - execution.modeled_impact_cost
        if bool((base_net_return <= -1.0).any().item()):
            raise RuntimeError("Net return reached -100% before forced fallback; check execution costs.")
        provisional_equity = old_equity * (1.0 + base_net_return)
        provisional_peak = torch.maximum(old_peak_equity, provisional_equity)
        provisional_drawdown = (1.0 - provisional_equity / provisional_peak.clamp_min(1e-12)).clamp(
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
            (self.batch_size,), next_index >= self.data.horizon, dtype=torch.bool, device=self.data.device
        )
        truncated = torch.full(
            (self.batch_size,),
            next_index >= self.max_episode_steps and next_index < self.data.horizon,
            dtype=torch.bool,
            device=self.data.device,
        )
        fallback_required = terminated | risk_halt_triggered
        next_weights = drifted.clone()
        next_weights[fallback_required] = 0.0
        next_weights[fallback_required, self.cash_index] = 1.0
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
            raise RuntimeError("Net one-step reward reached -100%; check return data and execution costs.")
        self._equity = old_equity * (1.0 + rewards.total)
        self._peak_equity = torch.maximum(old_peak_equity, self._equity)
        self._drawdown = (1.0 - self._equity / self._peak_equity.clamp_min(1e-12)).clamp(
            min=0.0,
            max=1.0,
        )
        self._weights = next_weights
        base_turnover = self._one_way_turnover(target_weights, old_weights)
        fallback_turnover = self._one_way_turnover(next_weights, drifted)
        self._recent_turnover = base_turnover + fallback_turnover
        self._gross_exposure = self._weights[:, self._risky_mask].sum(dim=-1)
        self._risk_halted = next_risk_halted
        self._time_index = next_index
        self._done = terminated | truncated
        next_starts = torch.zeros(self.batch_size, dtype=torch.bool, device=self.data.device)
        self._observation = self._build_observation(time_index=next_index, episode_start=next_starts)
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
            "environment_index": torch.arange(
                self.batch_size, dtype=torch.long, device=self.data.device
            ),
            "decision_index": torch.full(
                (self.batch_size,), next_index - 1, dtype=torch.long, device=self.data.device
            ),
            "step_index": torch.full(
                (self.batch_size,), self._time_index, dtype=torch.long, device=self.data.device
            ),
            **{f"execution_{name}": value for name, value in execution.diagnostics.items()},
            **{
                f"liquidation_execution_{name}": value
                for name, value in fallback_execution.diagnostics.items()
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
