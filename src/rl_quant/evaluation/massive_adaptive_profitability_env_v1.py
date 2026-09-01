"""Chronological three-book environment for adaptive economic control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_runtime_protocol_v1 import (
    MassiveAdaptiveForecastRuntimeProtocol,
    MassiveAdaptiveInferencePlanRuntimeProtocol,
)
from rl_quant.evaluation.massive_adaptive_initial_book_authority_v1 import (
    build_massive_adaptive_initial_book_authority_v1,
)
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
)
from rl_quant.evaluation.massive_adaptive_economic_step_v1 import (
    MassiveAdaptiveEconomicStepV1,
    MassiveAdaptivePreparedStepV1,
    prepare_massive_adaptive_economic_step_v1,
    settle_massive_adaptive_economic_step_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
    MassiveAdaptivePortfolioDecisionV1,
    compile_massive_adaptive_portfolio_v1,
)
from rl_quant.execution.massive_adaptive_rl_compiler_control_v1 import (
    MassiveAdaptiveRLCompilerControlV1,
    compile_massive_adaptive_rl_control_v1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MassiveAdaptiveFillSourceV1,
)
from rl_quant.features.massive_economic_authority_v6 import (
    MassiveProviderEconomicArchiveAuthorityV6,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    MassiveAdaptiveRLActionV1,
    neutral_massive_adaptive_rl_action_v1,
)
from rl_quant.rl.massive_adaptive_rl_observation_v1 import (
    MassiveAdaptiveRLObservationV1,
    MassiveAdaptiveRLTrailingStateV1,
    build_massive_adaptive_rl_observation_v1,
)

MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SCHEMA = (
    "rl-quant.massive-adaptive-profitability-env-v1"
)
MASSIVE_ADAPTIVE_RL_TRANSITION_V1_SCHEMA = "rl-quant.massive-adaptive-rl-transition-v1"
MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SPEC_SHA256 = semantic_sha256(
    {
        "chronology": "continuous-decision-close-to-next-close",
        "books": ("strategy", "neutral", "benchmark"),
        "optimization_reward": "10000-times-strategy-minus-neutral-log-wealth",
        "reported_returns": "unpenalized-economic-log-wealth",
        "terminal_accounting": "liquidation-cost-adjustment-at-true-end-only",
        "rollout_boundary": "state-preserving-truncation",
        "forecast_refit_boundary": "state-carry-without-liquidation",
        "target_access": False,
    }
)


class MassiveAdaptiveProfitabilityEnvV1Error(ValueError):
    """The environment chronology, state, or economic transition differs."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitabilityEnvStateV1:
    chronology_cursor: int
    strategy_book: MassiveAdaptiveEconomicBookV1
    neutral_book: MassiveAdaptiveEconomicBookV1
    benchmark_book: MassiveAdaptiveEconomicBookV1
    previous_action: MassiveAdaptiveRLActionV1
    trailing_state: MassiveAdaptiveRLTrailingStateV1
    done: bool
    source_inventory_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for value in (
            self.strategy_book,
            self.neutral_book,
            self.benchmark_book,
            self.previous_action,
            self.trailing_state,
        ):
            value.validate()
        if (
            isinstance(self.chronology_cursor, bool)
            or self.chronology_cursor < 0
            or not isinstance(self.done, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive profitability environment state differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTransitionV1:
    observation_receipt_sha256: str
    action_receipt_sha256: str
    compiler_control: MassiveAdaptiveRLCompilerControlV1
    policy_decision: MassiveAdaptivePortfolioDecisionV1
    neutral_decision: MassiveAdaptivePortfolioDecisionV1
    economic_step: MassiveAdaptiveEconomicStepV1
    optimization_reward_basis_points: float
    strategy_active_log_return: float
    neutral_active_log_return: float
    incremental_rl_log_return: float
    strategy_terminal_liquidation_cost: float
    neutral_terminal_liquidation_cost: float
    benchmark_terminal_liquidation_cost: float
    strategy_liquidation_adjusted_equity: float
    neutral_liquidation_adjusted_equity: float
    benchmark_liquidation_adjusted_equity: float
    source_data_qualified: bool
    terminated: bool
    truncated: bool
    next_state_receipt_sha256: str
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_TRANSITION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        self.compiler_control.validate()
        self.policy_decision.validate()
        self.neutral_decision.validate()
        self.economic_step.validate()
        values = (
            self.optimization_reward_basis_points,
            self.strategy_active_log_return,
            self.neutral_active_log_return,
            self.incremental_rl_log_return,
            self.strategy_terminal_liquidation_cost,
            self.neutral_terminal_liquidation_cost,
            self.benchmark_terminal_liquidation_cost,
            self.strategy_liquidation_adjusted_equity,
            self.neutral_liquidation_adjusted_equity,
            self.benchmark_liquidation_adjusted_equity,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_TRANSITION_V1_SCHEMA
            or any(not math.isfinite(value) for value in values)
            or min(values[4:]) < 0.0
            or not isinstance(self.source_data_qualified, bool)
            or self.source_data_qualified != self.economic_step.source_data_qualified
            or not isinstance(self.terminated, bool)
            or (self.terminated and self.truncated)
            or (self.truncated and any(abs(value) > 1.0e-12 for value in values[4:7]))
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive RL transition differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _state(
    *,
    cursor: int,
    strategy_book: MassiveAdaptiveEconomicBookV1,
    neutral_book: MassiveAdaptiveEconomicBookV1,
    benchmark_book: MassiveAdaptiveEconomicBookV1,
    previous_action: MassiveAdaptiveRLActionV1,
    trailing_state: MassiveAdaptiveRLTrailingStateV1,
    done: bool,
    source_inventory_sha256: str,
) -> MassiveAdaptiveProfitabilityEnvStateV1:
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFITABILITY_ENV_V1_SCHEMA,
        "chronology_cursor": cursor,
        "strategy_book": strategy_book,
        "neutral_book": neutral_book,
        "benchmark_book": benchmark_book,
        "previous_action": previous_action,
        "trailing_state": trailing_state,
        "done": done,
        "source_inventory_sha256": source_inventory_sha256,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveProfitabilityEnvStateV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


class MassiveAdaptiveProfitabilityEnvV1:
    """Stateful environment; only a true chronology end terminates economics."""

    def __init__(
        self,
        *,
        forecast_archive: MassiveAdaptiveForecastRuntimeProtocol,
        calibration: MassiveAdaptiveForecastCalibrationV2,
        inference_plan: MassiveAdaptiveInferencePlanRuntimeProtocol,
        decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
        context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
        fill_source: MassiveAdaptiveFillSourceV1,
        daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
        identity_authority: PITSecurityUniverseAuthority,
        economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None,
        initial_capital: float,
        transaction_cost_basis_points: float = 20.0,
        maximum_fill_participation: float = 0.02,
        compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
    ) -> None:
        for value in (
            forecast_archive,
            calibration,
            inference_plan,
            fill_source,
            daily_input_authority,
            identity_authority,
        ):
            value.validate()
        if economic_event_archive is not None:
            economic_event_archive.validate()
        if (
            not math.isfinite(initial_capital)
            or initial_capital <= 0.0
            or not math.isfinite(transaction_cost_basis_points)
            or transaction_cost_basis_points < 0.0
            or not 0.0 < maximum_fill_participation <= 1.0
            or forecast_archive.runtime_rows is None
            or not forecast_archive.runtime_forecasts_replayed
            or not inference_plan.rows
        ):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive profitability environment configuration differs"
            )
        self.forecast_archive = forecast_archive
        self.calibration = calibration
        self.inference_plan = inference_plan
        all_forecasts = {
            row.decision_session_date: row for row in forecast_archive.runtime_rows
        }
        self.roots = {row.decision_session_date: row for row in decision_roots}
        self.contexts = {row.decision_session_date: row for row in context_origins}
        plan_dates = tuple(row.decision_session_date for row in inference_plan.rows)
        if any(date not in all_forecasts for date in plan_dates):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive environment forecast inventory omits a planned decision"
            )
        self.forecasts = {date: all_forecasts[date] for date in plan_dates}
        if set(self.roots) != set(plan_dates) or set(self.contexts) != set(plan_dates):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive environment inventories differ"
            )
        self.fill_source = fill_source
        self.daily_input_authority = daily_input_authority
        self.identity_authority = identity_authority
        self.economic_event_archive = economic_event_archive
        self.initial_capital = float(initial_capital)
        self.transaction_cost_basis_points = float(transaction_cost_basis_points)
        self.maximum_fill_participation = float(maximum_fill_participation)
        self.compiler_config = (
            compiler_config or MassiveAdaptivePortfolioCompilerConfigV1()
        )
        self.compiler_config.validate()
        self.economic_compatibility_receipt_sha256 = semantic_sha256(
            (
                fill_source.semantic_receipt_sha256,
                daily_input_authority.semantic_receipt_sha256,
                identity_authority.receipt_sha256,
                None
                if economic_event_archive is None
                else economic_event_archive.receipt_sha256,
                self.compiler_config.receipt_sha256,
                self.initial_capital,
                self.transaction_cost_basis_points,
                self.maximum_fill_participation,
                "pit-equal-weight-staged-entry-then-buy-and-drift",
            )
        )
        self.validation_context_receipt_sha256 = semantic_sha256(
            (
                forecast_archive.semantic_receipt_sha256,
                calibration.semantic_receipt_sha256,
                inference_plan.semantic_receipt_sha256,
                tuple(self.roots[date].semantic_receipt_sha256 for date in plan_dates),
                tuple(
                    self.contexts[date].semantic_receipt_sha256 for date in plan_dates
                ),
                fill_source.semantic_receipt_sha256,
                daily_input_authority.semantic_receipt_sha256,
                identity_authority.receipt_sha256,
                None
                if economic_event_archive is None
                else economic_event_archive.receipt_sha256,
                self.compiler_config.receipt_sha256,
                self.initial_capital,
                self.maximum_fill_participation,
                "pit-equal-weight-staged-entry-then-buy-and-drift",
            )
        )
        self.source_inventory_sha256 = semantic_sha256(
            (
                forecast_archive.semantic_receipt_sha256,
                calibration.semantic_receipt_sha256,
                inference_plan.semantic_receipt_sha256,
                fill_source.semantic_receipt_sha256,
                daily_input_authority.semantic_receipt_sha256,
                identity_authority.receipt_sha256,
                None
                if economic_event_archive is None
                else economic_event_archive.receipt_sha256,
                self.compiler_config.receipt_sha256,
                self.initial_capital,
                self.transaction_cost_basis_points,
                self.maximum_fill_participation,
            )
        )
        self._state: MassiveAdaptiveProfitabilityEnvStateV1 | None = None
        self._prepared: MassiveAdaptivePreparedStepV1 | None = None
        self._observation: MassiveAdaptiveRLObservationV1 | None = None

    def _prepare_observation(self) -> MassiveAdaptiveRLObservationV1:
        if self._state is None or self._state.done:
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "cannot prepare an observation from a terminal state"
            )
        row = self.inference_plan.rows[self._state.chronology_cursor]
        self._prepared = prepare_massive_adaptive_economic_step_v1(
            forecast_archive=self.forecast_archive,
            forecast_row=self.forecasts[row.decision_session_date],
            calibration=self.calibration,
            inference_row=row,
            decision_root=self.roots[row.decision_session_date],
            context_origin=self.contexts[row.decision_session_date],
            strategy_book=self._state.strategy_book,
            neutral_book=self._state.neutral_book,
            benchmark_book=self._state.benchmark_book,
            daily_input_authority=self.daily_input_authority,
            identity_authority=self.identity_authority,
        )
        self._observation = build_massive_adaptive_rl_observation_v1(
            prepared=self._prepared,
            previous_action=self._state.previous_action,
            trailing_state=self._state.trailing_state,
        )
        return self._observation

    def reset(
        self,
    ) -> tuple[MassiveAdaptiveRLObservationV1, dict[str, object]]:
        first_date = self.inference_plan.rows[0].decision_session_date
        initial = build_massive_adaptive_initial_book_authority_v1(
            decision_session_date=first_date,
            initial_capital=self.initial_capital,
            forecast_archive_receipt_sha256=self.forecast_archive.semantic_receipt_sha256,
            inference_plan_receipt_sha256=self.inference_plan.semantic_receipt_sha256,
            source_data_qualified=bool(
                getattr(self.forecast_archive, "development_forecast_authorized", False)
                or getattr(self.forecast_archive, "outer_forecast_authorized", False)
            ),
        )
        self._state = _state(
            cursor=0,
            strategy_book=initial.strategy_book,
            neutral_book=initial.neutral_book,
            benchmark_book=initial.benchmark_book,
            previous_action=neutral_massive_adaptive_rl_action_v1(),
            trailing_state=MassiveAdaptiveRLTrailingStateV1(),
            done=False,
            source_inventory_sha256=self.source_inventory_sha256,
        )
        observation = self._prepare_observation()
        return observation, {
            "state_receipt_sha256": self._state.semantic_receipt_sha256
        }

    @property
    def state(self) -> MassiveAdaptiveProfitabilityEnvStateV1:
        if self._state is None:
            raise MassiveAdaptiveProfitabilityEnvV1Error("environment is not reset")
        return self._state

    @property
    def current_observation(self) -> MassiveAdaptiveRLObservationV1:
        if self._observation is None:
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive environment has no current observation"
            )
        return self._observation

    def restore(self, state: MassiveAdaptiveProfitabilityEnvStateV1) -> None:
        state.validate()
        if (
            state.source_inventory_sha256 != self.source_inventory_sha256
            or state.chronology_cursor > len(self.inference_plan.rows)
            or state.done != (state.chronology_cursor == len(self.inference_plan.rows))
        ):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "restored adaptive environment state is incompatible"
            )
        self._state = state
        self._prepared = None
        self._observation = None
        if not state.done:
            self._prepare_observation()

    def restore_continuation(
        self, state: MassiveAdaptiveProfitabilityEnvStateV1
    ) -> None:
        """Carry books into a new causal forecast archive at the same close.

        The caller must hold a package-derived economic-continuity authority.
        This low-level method only performs the state rebinding and therefore
        grants no authority by itself.
        """

        state.validate()
        first_date = self.inference_plan.rows[0].decision_session_date
        books = (state.strategy_book, state.neutral_book, state.benchmark_book)
        if (
            not state.done
            or state.chronology_cursor <= 0
            or any(book.decision_session_date != first_date for book in books)
        ):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive continuation state is not aligned to the next decision"
            )
        self._state = _state(
            cursor=0,
            strategy_book=state.strategy_book,
            neutral_book=state.neutral_book,
            benchmark_book=state.benchmark_book,
            previous_action=state.previous_action,
            trailing_state=state.trailing_state,
            done=False,
            source_inventory_sha256=self.source_inventory_sha256,
        )
        self._prepared = None
        self._observation = None
        self._prepare_observation()

    def rollout_boundary_state(self) -> MassiveAdaptiveProfitabilityEnvStateV1:
        """Return an unchanged continuation; no economic liquidation occurs."""

        return self.state

    @staticmethod
    def _liquidation_cost(
        book: MassiveAdaptiveEconomicBookV1, basis_points: float
    ) -> float:
        return sum(row.market_value for row in book.holdings) * basis_points / 10_000.0

    def step(
        self,
        action: MassiveAdaptiveRLActionV1,
        *,
        frozen_control: MassiveAdaptiveRLCompilerControlV1 | None = None,
        frozen_decision: MassiveAdaptivePortfolioDecisionV1 | None = None,
        continue_economic_episode: bool = False,
    ) -> tuple[
        MassiveAdaptiveRLObservationV1 | None,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        action.validate()
        if self._state is None or self._prepared is None or self._observation is None:
            raise MassiveAdaptiveProfitabilityEnvV1Error("environment is not reset")
        if self._state.done:
            raise MassiveAdaptiveProfitabilityEnvV1Error("environment is terminal")
        if not isinstance(continue_economic_episode, bool):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive continuation flag is invalid"
            )
        if (frozen_control is None) != (frozen_decision is None):
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "frozen control and decision must be supplied together"
            )
        if frozen_control is None or frozen_decision is None:
            control, policy_decision = compile_massive_adaptive_rl_control_v1(
                inputs=self._prepared.strategy_compiler_inputs,
                config=self.compiler_config,
                action=action,
            )
        else:
            frozen_control.validate()
            frozen_decision.validate()
            if (
                frozen_control.action_receipt_sha256 != action.semantic_receipt_sha256
                or frozen_control.adjusted_input_receipt_sha256
                != frozen_decision.input_receipt_sha256
                or frozen_decision.decision_id != self._prepared.decision_session_date
            ):
                raise MassiveAdaptiveProfitabilityEnvV1Error(
                    "frozen target replay differs from the prepared chronology"
                )
            control = frozen_control
            policy_decision = frozen_decision
        neutral_decision = compile_massive_adaptive_portfolio_v1(
            self._prepared.neutral_compiler_inputs,
            config=self.compiler_config,
        )
        economic_step = settle_massive_adaptive_economic_step_v1(
            prepared=self._prepared,
            policy_decision=policy_decision,
            neutral_decision=neutral_decision,
            fill_source=self.fill_source,
            economic_event_archive=self.economic_event_archive,
            daily_input_authority=self.daily_input_authority,
            identity_authority=self.identity_authority,
            transaction_cost_basis_points=self.transaction_cost_basis_points,
            maximum_fill_participation=self.maximum_fill_participation,
            policy_action_receipt_sha256=action.semantic_receipt_sha256,
            policy_control_receipt_sha256=control.semantic_receipt_sha256,
            policy_control=control,
            frozen_targets_replayed=frozen_decision is not None,
        )
        next_cursor = self._state.chronology_cursor + 1
        local_end = next_cursor == len(self.inference_plan.rows)
        if continue_economic_episode and not local_end:
            raise MassiveAdaptiveProfitabilityEnvV1Error(
                "adaptive economic continuation is only valid at an archive end"
            )
        terminated = bool(local_end and not continue_economic_episode)
        truncated = bool(local_end and continue_economic_episode)
        strategy_cost = neutral_cost = benchmark_cost = 0.0
        strategy_equity = economic_step.strategy_posttrade_book.marked_equity
        neutral_equity = economic_step.neutral_posttrade_book.marked_equity
        benchmark_equity = economic_step.benchmark_posttrade_book.marked_equity
        strategy_active = economic_step.strategy_active_log_return
        neutral_active = economic_step.neutral_active_log_return
        incremental = economic_step.incremental_rl_log_return
        if terminated:
            strategy_cost = self._liquidation_cost(
                economic_step.strategy_posttrade_book,
                self.transaction_cost_basis_points,
            )
            neutral_cost = self._liquidation_cost(
                economic_step.neutral_posttrade_book,
                self.transaction_cost_basis_points,
            )
            benchmark_cost = self._liquidation_cost(
                economic_step.benchmark_posttrade_book,
                self.transaction_cost_basis_points,
            )
            strategy_equity -= strategy_cost
            neutral_equity -= neutral_cost
            benchmark_equity -= benchmark_cost
            if min(strategy_equity, neutral_equity, benchmark_equity) <= 0.0:
                raise MassiveAdaptiveProfitabilityEnvV1Error(
                    "terminal liquidation exhausted an economic book"
                )
            strategy_terminal_adjustment = math.log(
                strategy_equity / economic_step.strategy_posttrade_book.marked_equity
            )
            neutral_terminal_adjustment = math.log(
                neutral_equity / economic_step.neutral_posttrade_book.marked_equity
            )
            benchmark_terminal_adjustment = math.log(
                benchmark_equity / economic_step.benchmark_posttrade_book.marked_equity
            )
            strategy_active += (
                strategy_terminal_adjustment - benchmark_terminal_adjustment
            )
            neutral_active += (
                neutral_terminal_adjustment - benchmark_terminal_adjustment
            )
            incremental += strategy_terminal_adjustment - neutral_terminal_adjustment
        reward = 10_000.0 * incremental
        execution_rows = economic_step.strategy_execution.rows
        requested = sum(abs(row.requested_shares) for row in execution_rows)
        unfilled = sum(abs(row.unfilled_shares) for row in execution_rows)
        fill_fraction = (
            1.0 if requested == 0.0 else max(0.0, 1.0 - unfilled / requested)
        )
        buy = sum(
            row.executed_notional for row in execution_rows if row.filled_shares > 0.0
        )
        sell = sum(
            row.executed_notional for row in execution_rows if row.filled_shares < 0.0
        )
        turnover = max(buy, sell) / self._state.strategy_book.marked_equity
        prior_trailing = self._state.trailing_state
        strategy_history = (
            *prior_trailing.strategy_active_log_returns,
            strategy_active,
        )[-63:]
        incremental_history = (
            *prior_trailing.incremental_rl_log_returns,
            incremental,
        )[-63:]
        strategy_book = economic_step.strategy_posttrade_book
        drawdown = 1.0 - strategy_book.marked_equity / strategy_book.high_water_mark
        trailing = MassiveAdaptiveRLTrailingStateV1(
            strategy_active_log_returns=tuple(strategy_history),
            incremental_rl_log_returns=tuple(incremental_history),
            previous_realized_turnover=turnover,
            previous_fill_fraction=fill_fraction,
            previous_unfilled_fraction=0.0
            if requested == 0.0
            else min(unfilled / requested, 1.0),
            previous_capacity_utilization=fill_fraction,
            current_drawdown=max(0.0, min(drawdown, 1.0)),
        )
        self._state = _state(
            cursor=next_cursor,
            strategy_book=economic_step.strategy_posttrade_book,
            neutral_book=economic_step.neutral_posttrade_book,
            benchmark_book=economic_step.benchmark_posttrade_book,
            previous_action=action,
            trailing_state=trailing,
            done=local_end,
            source_inventory_sha256=self.source_inventory_sha256,
        )
        body = {
            "schema": MASSIVE_ADAPTIVE_RL_TRANSITION_V1_SCHEMA,
            "observation_receipt_sha256": self._observation.semantic_receipt_sha256,
            "action_receipt_sha256": action.semantic_receipt_sha256,
            "compiler_control": control,
            "policy_decision": policy_decision,
            "neutral_decision": neutral_decision,
            "economic_step": economic_step,
            "optimization_reward_basis_points": reward,
            "strategy_active_log_return": strategy_active,
            "neutral_active_log_return": neutral_active,
            "incremental_rl_log_return": incremental,
            "strategy_terminal_liquidation_cost": strategy_cost,
            "neutral_terminal_liquidation_cost": neutral_cost,
            "benchmark_terminal_liquidation_cost": benchmark_cost,
            "strategy_liquidation_adjusted_equity": strategy_equity,
            "neutral_liquidation_adjusted_equity": neutral_equity,
            "benchmark_liquidation_adjusted_equity": benchmark_equity,
            "source_data_qualified": economic_step.source_data_qualified,
            "terminated": terminated,
            "truncated": truncated,
            "next_state_receipt_sha256": self._state.semantic_receipt_sha256,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
            "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        }
        provisional = MassiveAdaptiveRLTransitionV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256="0" * 64,
        )
        transition = replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        )
        transition.validate()
        next_observation = None if local_end else self._prepare_observation()
        return (
            next_observation,
            reward,
            terminated,
            truncated,
            {
                "transition": transition,
                "state_receipt_sha256": self._state.semantic_receipt_sha256,
            },
        )


__all__ = [
    "MassiveAdaptiveProfitabilityEnvStateV1",
    "MassiveAdaptiveProfitabilityEnvV1",
    "MassiveAdaptiveProfitabilityEnvV1Error",
    "MassiveAdaptiveRLTransitionV1",
]
