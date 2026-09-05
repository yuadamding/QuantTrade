from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    build_massive_adaptive_forecast_calibration_v2,
)
from rl_quant.evaluation.massive_adaptive_initial_book_authority_v1 import (
    build_massive_adaptive_initial_book_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_execution_result_v1 import (
    execute_massive_adaptive_order_intent_v1,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
    MassiveAdaptiveRLTransitionV1,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_v1 import (
    replay_massive_adaptive_rl_frozen_target_transitions_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_authority_v2 import (
    MassiveAdaptiveRLOuterRolloutComputationV2,
    _decision_target_inventory,
    _terminal_adjusted_rows,
)
from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v2 import (
    MassiveAdaptiveRLProfitabilityFoldReportV2,
    _cost_ladder_monotonicity_gate,
)
from rl_quant.execution.massive_adaptive_order_intent_v1 import (
    build_massive_adaptive_target_order_intent_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import adaptive_fill_clock_v1
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptiveBoundedControlDistributionV1,
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import (
    MassiveAdaptiveRLActionV1,
    build_massive_adaptive_rl_action_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPOTrainerV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    registered_massive_adaptive_rl_constant_actions_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
)
from rl_quant.workflows.massive_adaptive_rl_prequential_experiment_state_v1 import (
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1,
)


_SESSION_COUNT = 126
_SECURITY_COUNT = 4
_INITIAL_CAPITAL = 1_000_000.0


def _digest(value: object) -> str:
    return semantic_sha256(("v5-real-vertical", value))


def _business_dates(start: date, count: int) -> tuple[str, ...]:
    rows: list[str] = []
    cursor = start
    while len(rows) < count:
        if cursor.weekday() < 5:
            rows.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(rows)


def _identity(security_ids: tuple[str, ...]) -> PITSecurityUniverseAuthority:
    rule = PITUniverseRuleSpec.build(
        rule_id="adaptive-v5-real-vertical",
        target_size=len(security_ids),
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )
    masters = tuple(
        SourcedSecurityMasterRecord(
            security_id=security_id,
            issuer_id=f"ISSUER-{index:03d}",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=1,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id=f"CHAIN-{index:03d}",
            identity_source_receipt_sha256=_digest((security_id, "master")),
        )
        for index, security_id in enumerate(security_ids)
    )
    tickers = tuple(
        SourcedTickerHistoryRecord(
            security_id=security_id,
            ticker=f"T{index:03d}",
            valid_from_ms=1,
            valid_to_ms=None,
            available_at_ms=1,
            primary_exchange="XNYS",
            source_receipt_sha256=_digest((security_id, "ticker")),
        )
        for index, security_id in enumerate(security_ids)
    )
    listings = tuple(
        ListingEventRecord(
            event_id=f"LIST-{security_id}",
            security_id=security_id,
            effective_at_ms=1,
            available_at_ms=1,
            primary_exchange="XNYS",
            ticker=f"T{index:03d}",
            source_receipt_sha256=_digest((security_id, "listing")),
        )
        for index, security_id in enumerate(security_ids)
    )
    ranks = tuple(
        UniverseRankInputRecord(
            security_id=security_id,
            effective_at_ms=100,
            effective_session_index=10,
            available_at_ms=99,
            observation_start_ms=1,
            observation_end_ms=98,
            observation_start_session_index=7,
            observation_end_session_index=9,
            observed_session_count=3,
            average_dollar_volume=100_000_000.0,
            close_price=100.0,
            source_receipt_sha256=_digest((security_id, "rank")),
        )
        for security_id in security_ids
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=ranks,
    )


def _calibration(security_ids: tuple[str, ...], fold_index: int):
    training_date = "2019-12-31"
    window_receipt = _digest((fold_index, "training-window"))
    checkpoint_receipt = _digest((fold_index, "supervised-checkpoint"))
    model_state_receipt = _digest((fold_index, "supervised-model-state"))
    forecast_row = SimpleNamespace(
        decision_session_date=training_date,
        security_ids=security_ids,
        residual_mean=torch.zeros((len(security_ids), 7), dtype=torch.float32),
        residual_scale=torch.full((len(security_ids), 7), 0.01, dtype=torch.float32),
        valid=torch.ones(len(security_ids), dtype=torch.bool),
        receipt_sha256=_digest((fold_index, "training-forecast-row")),
    )
    target_rows = tuple(
        SimpleNamespace(
            security_id=security_id,
            residual_bucket_returns=tuple(
                0.01 * (bucket + 1) * (1.0 + index / len(security_ids))
                for bucket in range(7)
            ),
            training_valid_by_bucket=(True,) * 7,
            receipt_sha256=_digest((fold_index, "training-target", security_id)),
        )
        for index, security_id in enumerate(security_ids)
    )
    source_target = SimpleNamespace(
        decision_session_date=training_date,
        targets=SimpleNamespace(security_ids=security_ids, rows=target_rows),
    )
    training_forecasts = SimpleNamespace(
        validate=lambda: None,
        runtime_rows=(forecast_row,),
        runtime_forecasts_replayed=True,
        origin_session_dates=(training_date,),
        semantic_receipt_sha256=_digest((fold_index, "training-forecasts")),
        committed_source_data_qualified=False,
        window_plan_receipt_sha256=window_receipt,
        checkpoint_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state_receipt,
    )
    training_targets = SimpleNamespace(
        validate=lambda: None,
        runtime_source_targets=(source_target,),
        runtime_roots_replayed=True,
        decision_session_dates=(training_date,),
        semantic_receipt_sha256=_digest((fold_index, "training-targets")),
        committed_source_data_qualified=False,
    )
    training_window = SimpleNamespace(
        validate=lambda: None,
        split_role="training",
        fold_index=fold_index,
        semantic_receipt_sha256=window_receipt,
        rows=(SimpleNamespace(origin_session_date=training_date),),
    )
    checkpoint = SimpleNamespace(
        validate=lambda: None,
        window_plan_receipt_sha256=window_receipt,
        semantic_receipt_sha256=checkpoint_receipt,
        model_state_receipt_sha256=model_state_receipt,
        loaded_source=SimpleNamespace(
            receipt_sha256=_digest((fold_index, "checkpoint-source"))
        ),
        development_training_authorized=False,
    )
    return build_massive_adaptive_forecast_calibration_v2(
        checkpoint=checkpoint,
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window,
    )


@dataclass(frozen=True)
class _MarketFixture:
    fold_index: int
    dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    forecast_archive: object
    calibration: object
    inference_plan: object
    roots: tuple[object, ...]
    contexts: tuple[object, ...]
    fill_source: object
    daily: object
    identity: PITSecurityUniverseAuthority

    def environment(
        self, cost_basis_points: float
    ) -> MassiveAdaptiveProfitabilityEnvV1:
        return MassiveAdaptiveProfitabilityEnvV1(
            forecast_archive=self.forecast_archive,
            calibration=self.calibration,
            inference_plan=self.inference_plan,
            decision_roots=self.roots,
            context_origins=self.contexts,
            fill_source=self.fill_source,
            daily_input_authority=self.daily,
            identity_authority=self.identity,
            economic_event_archive=None,
            initial_capital=_INITIAL_CAPITAL,
            transaction_cost_basis_points=cost_basis_points,
            compiler_config=MassiveAdaptivePortfolioCompilerConfigV1(
                maximum_security_weight=0.25,
                maximum_issuer_weight=0.25,
                tracking_error_limit_annualized=0.50,
                absolute_active_beta_limit=1.0,
                maximum_daily_one_way_turnover=0.50,
                solver_step_size=0.10,
                solver_max_iterations=100,
                projection_max_iterations=100,
                numerical_tolerance=1.0e-2,
            ),
        )


def _market_fixture(
    fold_index: int, *, session_count: int = _SESSION_COUNT
) -> _MarketFixture:
    all_dates = _business_dates(date(2020 + fold_index * 2, 1, 2), session_count + 1)
    decision_dates = all_dates[:-1]
    security_ids = tuple(f"SEC-{index:03d}" for index in range(_SECURITY_COUNT))
    identity = _identity(security_ids)
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    bars_rows: dict[tuple[str, str], object] = {}
    for date_index, session_date in enumerate(all_dates):
        for asset_index, security_id in enumerate(security_ids):
            values = [0.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            direction = 1.0 if asset_index < len(security_ids) // 2 else -1.0
            cycle = ((date_index + fold_index) % 9 - 4) * 0.015
            price = 100.0 + asset_index * 0.25 + direction * 0.04 * date_index + cycle
            values[close_index] = price
            values[dollar_index] = 100_000_000.0
            bars_rows[(session_date, security_id)] = SimpleNamespace(
                bars_values=tuple(values),
                bars_valid=(True,) * len(values),
                receipt_sha256=_digest(
                    (fold_index, "daily", session_date, security_id)
                ),
            )
    daily_receipt = _digest((fold_index, "daily-input"))
    daily = SimpleNamespace(
        validate=lambda: None,
        sessions=tuple(
            SimpleNamespace(
                source_session_date=value,
                regular_close_at_ms=adaptive_fill_clock_v1(value)[1]
                + 6 * 60 * 60 * 1_000,
            )
            for value in all_dates
        ),
        row=lambda *, session_date, security_id: bars_rows[(session_date, security_id)],
        semantic_receipt_sha256=daily_receipt,
        daily_input_data_qualified=False,
    )
    fill_rows: dict[tuple[str, str], object] = {}
    for date_index, session_date in enumerate(all_dates[1:], start=1):
        for asset_index, security_id in enumerate(security_ids):
            prior = bars_rows[(all_dates[date_index - 1], security_id)].bars_values[
                close_index
            ]
            current = bars_rows[(session_date, security_id)].bars_values[close_index]
            price = float((prior + current) / 2.0)
            fill_rows[(session_date, security_id)] = SimpleNamespace(
                valid=True,
                fill_vwap=price,
                qualifying_share_volume=10_000_000.0,
                receipt_sha256=_digest((fold_index, "fill", session_date, security_id)),
            )
    fill_source = SimpleNamespace(
        validate=lambda: None,
        row=lambda *, session_date, security_id: fill_rows[(session_date, security_id)],
        daily_input_authority_semantic_receipt_sha256=daily_receipt,
        semantic_receipt_sha256=_digest((fold_index, "fill-source")),
        source_data_qualified=False,
    )
    forecasts: list[object] = []
    roots: list[object] = []
    contexts: list[object] = []
    plan_rows: list[object] = []
    for date_index, (session_date, next_session) in enumerate(
        zip(decision_dates, all_dates[1:], strict=True)
    ):
        root_receipt = _digest((fold_index, "root", session_date))
        context_receipt = _digest((fold_index, "context", session_date))
        inference_receipt = _digest((fold_index, "inference", session_date))
        means = torch.empty((len(security_ids), 7), dtype=torch.float32)
        for asset_index in range(len(security_ids)):
            signal = 0.02 if asset_index < len(security_ids) // 2 else -0.01
            signal += 0.0005 * math.sin(date_index + asset_index)
            means[asset_index] = signal * torch.arange(1, 8, dtype=torch.float32)
        forecasts.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                security_ids=security_ids,
                residual_mean=means,
                residual_scale=torch.full_like(means, 0.002),
                valid=torch.ones(len(security_ids), dtype=torch.bool),
                decision_root_receipt_sha256=root_receipt,
                inference_row_receipt_sha256=inference_receipt,
                array_receipts=tuple(
                    _digest((fold_index, session_date, index)) for index in range(18)
                ),
                receipt_sha256=_digest((fold_index, "forecast-row", session_date)),
            )
        )
        roots.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                context_security_ids=security_ids,
                semantic_receipt_sha256=root_receipt,
                context_origin_receipt_sha256=context_receipt,
                source_data_qualified=False,
            )
        )
        contexts.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                semantic_receipt_sha256=context_receipt,
                identity_authority_receipt_sha256=identity.receipt_sha256,
            )
        )
        plan_rows.append(
            SimpleNamespace(
                validate=lambda **_: None,
                decision_session_date=session_date,
                next_session_date=next_session,
                context_session_dates=(session_date,),
                receipt_sha256=inference_receipt,
            )
        )
    calibration = _calibration(security_ids, fold_index)
    forecast_archive = SimpleNamespace(
        validate=lambda: None,
        fold_index=fold_index,
        checkpoint_receipt_sha256=calibration.checkpoint_receipt_sha256,
        model_state_receipt_sha256=calibration.model_state_receipt_sha256,
        training_window_plan_receipt_sha256=(
            calibration.training_window_plan_receipt_sha256
        ),
        runtime_rows=tuple(forecasts),
        runtime_forecasts_replayed=True,
        row_receipts=tuple(row.receipt_sha256 for row in forecasts),
        semantic_receipt_sha256=_digest((fold_index, "forecast-archive")),
        development_forecast_authorized=False,
        outer_forecast_authorized=False,
    )
    inference_plan = SimpleNamespace(
        validate=lambda: None,
        inference_role="outer_test",
        fold_index=fold_index,
        rows=tuple(plan_rows),
        semantic_receipt_sha256=_digest((fold_index, "inference-plan")),
    )
    return _MarketFixture(
        fold_index=fold_index,
        dates=decision_dates,
        security_ids=security_ids,
        forecast_archive=forecast_archive,
        calibration=calibration,
        inference_plan=inference_plan,
        roots=tuple(roots),
        contexts=tuple(contexts),
        fill_source=fill_source,
        daily=daily,
        identity=identity,
    )


@dataclass(frozen=True)
class _ActionEvidence:
    decision_session_date: str
    observation_receipt_sha256: str
    action_values: tuple[float, ...]


def _rollout(
    environment: MassiveAdaptiveProfitabilityEnvV1,
    *,
    model: MassiveAdaptivePPOActorCriticV1 | None = None,
    action: MassiveAdaptiveRLActionV1 | None = None,
) -> tuple[tuple[MassiveAdaptiveRLTransitionV1, ...], tuple[_ActionEvidence, ...]]:
    observation, _ = environment.reset()
    transitions: list[MassiveAdaptiveRLTransitionV1] = []
    evidence: list[_ActionEvidence] = []
    while True:
        if model is not None:
            with torch.inference_mode():
                output = model(
                    {
                        "adaptive_state": torch.tensor(
                            observation.values, dtype=torch.float32
                        ).unsqueeze(0)
                    }
                )
            distribution = output.distribution
            assert isinstance(distribution, MassiveAdaptiveBoundedControlDistributionV1)
            values = tuple(
                float(value)
                for value in distribution.deterministic_action()[0].tolist()
            )
            selected_action = build_massive_adaptive_rl_action_v1(
                bucket_controls=values[:7],
                uncertainty_control=values[7],
                risk_control=values[8],
                trade_cost_control=values[9],
            )
        else:
            assert action is not None
            selected_action = action
            values = (
                *selected_action.bucket_controls,
                selected_action.uncertainty_control,
                selected_action.risk_control,
                selected_action.trade_cost_control,
            )
        decision_date = environment.inference_plan.rows[
            environment.state.chronology_cursor
        ].decision_session_date
        evidence.append(
            _ActionEvidence(
                decision_session_date=decision_date,
                observation_receipt_sha256=observation.semantic_receipt_sha256,
                action_values=values,
            )
        )
        next_observation, _, terminated, truncated, info = environment.step(
            selected_action
        )
        assert not truncated
        transition = info["transition"]
        assert isinstance(transition, MassiveAdaptiveRLTransitionV1)
        transitions.append(transition)
        if terminated:
            break
        assert next_observation is not None
        observation = next_observation
    return tuple(transitions), tuple(evidence)


def _terminal_return(
    rows: tuple[MassiveAdaptiveRLTransitionV1, ...], book: str
) -> float:
    return math.expm1(sum(_terminal_adjusted_rows(rows, book=book)))


def _drawdown(log_returns: tuple[float, ...]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum = 0.0
    for value in log_returns:
        wealth *= math.exp(value)
        peak = max(peak, wealth)
        maximum = max(maximum, 1.0 - wealth / peak)
    return maximum


@dataclass(frozen=True)
class _FoldVertical:
    fold_index: int
    model_updated: bool
    checkpoint_receipt: str
    computation: MassiveAdaptiveRLOuterRolloutComputationV2


def _run_fold(fold_index: int) -> _FoldVertical:
    market = _market_fixture(fold_index)
    training_environment = market.environment(20.0)
    model = MassiveAdaptivePPOActorCriticV1(observation_dim=90, hidden_dim=32)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    trainer = MassiveAdaptivePPOTrainerV1(
        environment=training_environment,
        model=model,
        config=MassiveAdaptivePPOConfigV1(
            rollout_length=8,
            minibatch_size=8,
            epochs_per_rollout=1,
            seed=17 + fold_index,
        ),
    )
    rollout = trainer.collect_rollout()
    trainer.update(rollout)
    checkpoint = trainer.checkpoint()
    model_updated = any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )

    primary, evidence = _rollout(market.environment(20.0), model=model)
    low = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=evidence,
        primary_transitions=primary,
        environment=market.environment(10.0),
    )
    high = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=evidence,
        primary_transitions=primary,
        environment=market.environment(40.0),
    )
    fixed_action = dict(registered_massive_adaptive_rl_constant_actions_v1())["FC02"]
    fixed, fixed_evidence = _rollout(market.environment(20.0), action=fixed_action)
    fixed_low = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=fixed_evidence,
        primary_transitions=fixed,
        environment=market.environment(10.0),
    )
    fixed_high = replay_massive_adaptive_rl_frozen_target_transitions_v1(
        primary_action_evidence=fixed_evidence,
        primary_transitions=fixed,
        environment=market.environment(40.0),
    )
    strategy = _terminal_adjusted_rows(primary, book="strategy")
    neutral = _terminal_adjusted_rows(primary, book="neutral")
    benchmark = _terminal_adjusted_rows(primary, book="benchmark")
    fixed_series = _terminal_adjusted_rows(fixed, book="strategy")
    primary_return = math.expm1(sum(strategy))
    low_return = _terminal_return(low, "strategy")
    high_return = _terminal_return(high, "strategy")
    fixed_return = math.expm1(sum(fixed_series))
    fixed_low_return = _terminal_return(fixed_low, "strategy")
    fixed_high_return = _terminal_return(fixed_high, "strategy")
    action_receipts = tuple(
        _digest((fold_index, "action", index, row.action_values))
        for index, row in enumerate(evidence)
    )
    transition_groups = (primary, low, high, fixed, fixed_low, fixed_high)
    transition_inventories = tuple(
        semantic_sha256(tuple(row.semantic_receipt_sha256 for row in rows))
        for rows in transition_groups
    )
    body = {
        "fold_index": fold_index,
        "outer_access_commitment_receipt_sha256": _digest((fold_index, "outer-access")),
        "frozen_ppo_policy_receipt_sha256": _digest((fold_index, "frozen-ppo")),
        "frozen_fc06_control_receipt_sha256": _digest((fold_index, "frozen-fc")),
        "decision_session_dates": market.dates,
        "ppo_action_evidence_receipts": action_receipts,
        "ppo_action_inventory_sha256": semantic_sha256(action_receipts),
        "ppo_primary_trace_receipt_sha256": _digest((fold_index, "primary-trace")),
        "ppo_low_cost_trace_receipt_sha256": _digest((fold_index, "low-trace")),
        "ppo_high_cost_trace_receipt_sha256": _digest((fold_index, "high-trace")),
        "fixed_control_trace_receipt_sha256": _digest((fold_index, "fixed-trace")),
        "fixed_control_low_cost_trace_receipt_sha256": _digest(
            (fold_index, "fixed-low-trace")
        ),
        "fixed_control_high_cost_trace_receipt_sha256": _digest(
            (fold_index, "fixed-high-trace")
        ),
        "ppo_primary_transition_inventory_sha256": transition_inventories[0],
        "ppo_low_cost_transition_inventory_sha256": transition_inventories[1],
        "ppo_high_cost_transition_inventory_sha256": transition_inventories[2],
        "fixed_control_transition_inventory_sha256": transition_inventories[3],
        "fixed_control_low_cost_transition_inventory_sha256": (
            transition_inventories[4]
        ),
        "fixed_control_high_cost_transition_inventory_sha256": (
            transition_inventories[5]
        ),
        "decision_target_inventory_sha256": _decision_target_inventory(primary),
        "fixed_control_decision_target_inventory_sha256": (
            _decision_target_inventory(fixed)
        ),
        "strategy_net_log_returns": strategy,
        "neutral_net_log_returns": neutral,
        "benchmark_net_log_returns": benchmark,
        "fixed_control_net_log_returns": fixed_series,
        "active_log_returns": tuple(
            left - right for left, right in zip(strategy, benchmark, strict=True)
        ),
        "incremental_rl_log_returns": tuple(
            left - right for left, right in zip(strategy, neutral, strict=True)
        ),
        "ppo_minus_fixed_control_log_returns": tuple(
            left - right for left, right in zip(strategy, fixed_series, strict=True)
        ),
        "primary_terminal_liquidation_adjusted_return": primary_return,
        "low_cost_terminal_liquidation_adjusted_return": low_return,
        "high_cost_terminal_liquidation_adjusted_return": high_return,
        "fixed_control_terminal_liquidation_adjusted_return": fixed_return,
        "fixed_control_low_cost_terminal_liquidation_adjusted_return": (
            fixed_low_return
        ),
        "fixed_control_high_cost_terminal_liquidation_adjusted_return": (
            fixed_high_return
        ),
        "ppo_cost_ladder_monotone": low_return >= primary_return >= high_return,
        "fixed_control_cost_ladder_monotone": (
            fixed_low_return >= fixed_return >= fixed_high_return
        ),
        "maximum_drawdown": _drawdown(strategy),
        "environment_source_inventory_sha256": _digest(
            (fold_index, market.forecast_archive.semantic_receipt_sha256)
        ),
        "source_data_qualified": False,
    }
    provisional = MassiveAdaptiveRLOuterRolloutComputationV2(
        **body,
        semantic_receipt_sha256="0" * 64,
        _ppo_primary_transitions=primary,
        _ppo_low_cost_transitions=low,
        _ppo_high_cost_transitions=high,
        _fixed_control_transitions=fixed,
        _fixed_control_low_cost_transitions=fixed_low,
        _fixed_control_high_cost_transitions=fixed_high,
    )
    computation = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    computation.validate()
    return _FoldVertical(
        fold_index=fold_index,
        model_updated=model_updated,
        checkpoint_receipt=checkpoint.semantic_receipt_sha256,
        computation=computation,
    )


@pytest.fixture(scope="module")
def real_vertical() -> tuple[_FoldVertical, ...]:
    return tuple(_run_fold(fold_index) for fold_index in range(4))


def _fold_report(computation: MassiveAdaptiveRLOuterRolloutComputationV2):
    rows = computation.strategy_net_log_returns
    body = {
        "fold_index": computation.fold_index,
        "outer_fold_seal_receipt_sha256": _digest((computation.fold_index, "seal")),
        "outer_rollout_authority_receipt_sha256": computation.semantic_receipt_sha256,
        "decision_session_dates": computation.decision_session_dates,
        "strategy_net_log_returns": rows,
        "benchmark_net_log_returns": computation.benchmark_net_log_returns,
        "neutral_net_log_returns": computation.neutral_net_log_returns,
        "fixed_control_net_log_returns": computation.fixed_control_net_log_returns,
        "active_log_returns": computation.active_log_returns,
        "incremental_rl_log_returns": computation.incremental_rl_log_returns,
        "ppo_minus_fixed_control_log_returns": (
            computation.ppo_minus_fixed_control_log_returns
        ),
        "cumulative_net_log_return": sum(rows),
        "terminal_liquidation_adjusted_return": (
            computation.primary_terminal_liquidation_adjusted_return
        ),
        "low_cost_terminal_liquidation_adjusted_return": (
            computation.low_cost_terminal_liquidation_adjusted_return
        ),
        "high_cost_terminal_liquidation_adjusted_return": (
            computation.high_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_terminal_liquidation_adjusted_return": (
            computation.fixed_control_terminal_liquidation_adjusted_return
        ),
        "fixed_control_low_cost_terminal_liquidation_adjusted_return": (
            computation.fixed_control_low_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_high_cost_terminal_liquidation_adjusted_return": (
            computation.fixed_control_high_cost_terminal_liquidation_adjusted_return
        ),
        "ppo_cost_ladder_monotone": computation.ppo_cost_ladder_monotone,
        "fixed_control_cost_ladder_monotone": (
            computation.fixed_control_cost_ladder_monotone
        ),
        "annualized_net_return": math.expm1(252.0 * sum(rows) / len(rows)),
        "annualized_volatility": torch.tensor(rows).std(unbiased=True).item()
        * math.sqrt(252.0),
        "net_sharpe_ratio": 0.0,
        "maximum_drawdown": computation.maximum_drawdown,
        "source_data_qualified": False,
    }
    provisional = MassiveAdaptiveRLProfitabilityFoldReportV2(
        **body, semantic_receipt_sha256="0" * 64
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def test_one_step_position_return_lag(real_vertical) -> None:
    transitions = real_vertical[0].computation.ppo_primary_transitions
    first, second = transitions[:2]
    assert first.economic_step.strategy_execution.fill_session_date > (
        first.economic_step.strategy_execution.decision_session_date
    )
    assert second.economic_step.strategy_execution.pretrade_book_receipt_sha256 == (
        first.economic_step.strategy_posttrade_book.semantic_receipt_sha256
    )


def test_unchanged_position_has_zero_turnover_cost() -> None:
    market = _market_fixture(0, session_count=2)
    first_date, next_date = (
        market.dates[0],
        market.inference_plan.rows[0].next_session_date,
    )
    initial = build_massive_adaptive_initial_book_authority_v1(
        decision_session_date=first_date,
        initial_capital=_INITIAL_CAPITAL,
        forecast_archive_receipt_sha256=(
            market.forecast_archive.semantic_receipt_sha256
        ),
        inference_plan_receipt_sha256=market.inference_plan.semantic_receipt_sha256,
        source_data_qualified=False,
    )
    target_receipt = _digest("unchanged-cash-target")
    intent = build_massive_adaptive_target_order_intent_v1(
        decision_session_date=first_date,
        scheduled_fill_session_date=next_date,
        book=initial.strategy_book,
        security_ids=(),
        target_weights=(),
        target_receipt_sha256=target_receipt,
        decision_marks={},
        decision_mark_receipts={},
    )
    result = execute_massive_adaptive_order_intent_v1(
        order_intent=intent,
        book=initial.strategy_book,
        fill_source=market.fill_source,
        daily_input_authority=market.daily,
        identity_authority=market.identity,
        transaction_cost_basis_points=40.0,
    )
    assert result.gross_traded_notional == 0.0
    assert result.total_transaction_cost == 0.0
    assert result.posttrade_book.cash == _INITIAL_CAPITAL


def test_nonmonotone_fixed_target_cost_ladder_is_reported_as_failed_gate(
    real_vertical,
) -> None:
    fold_reports = tuple(_fold_report(row.computation) for row in real_vertical)
    first = fold_reports[0]
    nonmonotone = replace(
        first,
        low_cost_terminal_liquidation_adjusted_return=(
            first.terminal_liquidation_adjusted_return - 0.001
        ),
        ppo_cost_ladder_monotone=False,
        semantic_receipt_sha256="0" * 64,
    )
    nonmonotone = replace(
        nonmonotone,
        semantic_receipt_sha256=semantic_sha256(nonmonotone.semantic_unsigned()),
    )
    nonmonotone.validate()
    assert not _cost_ladder_monotonicity_gate((nonmonotone, *fold_reports[1:]))


def test_terminal_liquidation_compounding_identity(real_vertical) -> None:
    for row in real_vertical:
        computation = row.computation
        prior_cash = _INITIAL_CAPITAL
        prior_equity = _INITIAL_CAPITAL
        prior_shares: dict[str, float] = {}
        for transition in computation.ppo_primary_transitions:
            execution = transition.economic_step.strategy_execution
            expected_cash = prior_cash
            expected_shares = dict(prior_shares)
            expected_cost = 0.0
            expected_notional = 0.0
            for fill in execution.rows:
                expected_row_notional = abs(fill.filled_shares) * fill.fill_price
                expected_row_cost = expected_row_notional * 20.0 / 10_000.0
                assert math.isclose(
                    fill.executed_notional,
                    expected_row_notional,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                assert math.isclose(
                    fill.transaction_cost,
                    expected_row_cost,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                expected_cash -= (
                    fill.filled_shares * fill.fill_price + expected_row_cost
                )
                expected_shares[fill.security_id] = (
                    expected_shares.get(fill.security_id, 0.0) + fill.filled_shares
                )
                if expected_shares[fill.security_id] <= 1.0e-12:
                    expected_shares.pop(fill.security_id)
                expected_cost += expected_row_cost
                expected_notional += expected_row_notional

            book = execution.posttrade_book
            observed_shares = book.shares_by_security()
            assert observed_shares.keys() == expected_shares.keys()
            for security_id, expected_quantity in expected_shares.items():
                assert math.isclose(
                    observed_shares[security_id],
                    expected_quantity,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
            assert math.isclose(
                execution.total_transaction_cost,
                expected_cost,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            assert math.isclose(
                execution.gross_traded_notional,
                expected_notional,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            assert math.isclose(
                book.cash,
                expected_cash,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
            independently_marked_equity = book.cash + sum(
                holding.shares * holding.last_mark for holding in book.holdings
            )
            assert math.isclose(
                book.marked_equity,
                independently_marked_equity,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
            assert math.isclose(
                transition.economic_step.strategy_net_log_return,
                math.log(independently_marked_equity / prior_equity),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            prior_cash = book.cash
            prior_equity = book.marked_equity
            prior_shares = observed_shares

        terminal = computation.ppo_primary_transitions[-1]
        terminal_book = terminal.economic_step.strategy_posttrade_book
        expected_liquidation_cost = (
            sum(holding.market_value for holding in terminal_book.holdings)
            * 20.0
            / 10_000.0
        )
        expected_terminal_equity = (
            terminal_book.marked_equity - expected_liquidation_cost
        )
        assert math.isclose(
            terminal.strategy_terminal_liquidation_cost,
            expected_liquidation_cost,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        )
        assert math.isclose(
            terminal.strategy_liquidation_adjusted_equity,
            expected_terminal_equity,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
        assert math.isclose(
            computation.primary_terminal_liquidation_adjusted_return,
            expected_terminal_equity / _INITIAL_CAPITAL - 1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
        assert math.isclose(
            math.expm1(sum(computation.strategy_net_log_returns)),
            computation.primary_terminal_liquidation_adjusted_return,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )


def test_ppo_fc06_and_benchmark_share_outer_economics(real_vertical) -> None:
    for row in real_vertical:
        computation = row.computation
        primary = computation.ppo_primary_transitions
        fixed = computation._fixed_control_transitions
        assert tuple(
            value.economic_step.strategy_execution.fill_source_receipt_sha256
            for value in primary
        ) == tuple(
            value.economic_step.strategy_execution.fill_source_receipt_sha256
            for value in fixed
        )
        assert tuple(
            value.economic_step.strategy_execution.daily_input_receipt_sha256
            for value in primary
        ) == tuple(
            value.economic_step.strategy_execution.daily_input_receipt_sha256
            for value in fixed
        )
        assert len(computation.benchmark_net_log_returns) == _SESSION_COUNT


def test_outer_zero_seal_precedes_validation_two_release() -> None:
    sequence = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1
    assert sequence.index("outer-0-sealed") < sequence.index("validation-2-released")


def test_outer_one_seal_precedes_validation_three_release() -> None:
    sequence = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1
    assert sequence.index("outer-1-sealed") < sequence.index("validation-3-released")
    assert sequence.index("outer-2-sealed") < sequence.index("outer-3-sealed")


def test_diagnostic_schedule_completes_outer_report(real_vertical) -> None:
    reports = tuple(_fold_report(row.computation) for row in real_vertical)
    assert len(reports) == 4
    assert tuple(row.fold_index for row in reports) == (0, 1, 2, 3)
    diagnostic_schedule_qualified = False
    profitability_gates_passed = _cost_ladder_monotonicity_gate(reports)
    positive_authorization = bool(
        diagnostic_schedule_qualified and profitability_gates_passed
    )
    assert not positive_authorization


def test_every_stage_resumes_to_identical_receipts() -> None:
    market = _market_fixture(0, session_count=8)
    config = MassiveAdaptivePPOConfigV1(
        rollout_length=4,
        minibatch_size=4,
        epochs_per_rollout=1,
        seed=71,
    )
    first = MassiveAdaptivePPOTrainerV1(
        environment=market.environment(20.0),
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90, hidden_dim=32),
        config=config,
    )
    first.update(first.collect_rollout())
    checkpoint = first.checkpoint()
    expected_rollout = first.collect_rollout()
    expected_metrics = first.update(expected_rollout)
    expected_checkpoint = first.checkpoint()

    resumed = MassiveAdaptivePPOTrainerV1(
        environment=market.environment(20.0),
        model=MassiveAdaptivePPOActorCriticV1(observation_dim=90, hidden_dim=32),
        config=config,
    )
    resumed.restore(checkpoint)
    resumed_rollout = resumed.collect_rollout()
    resumed_metrics = resumed.update(resumed_rollout)
    resumed_checkpoint = resumed.checkpoint()
    assert resumed_rollout.transition_receipts == expected_rollout.transition_receipts
    assert resumed_metrics == expected_metrics
    assert resumed_checkpoint.semantic_receipt_sha256 == (
        expected_checkpoint.semantic_receipt_sha256
    )


def test_predecessor_tampering_blocks_authorization(real_vertical) -> None:
    computation = real_vertical[0].computation
    tampered = replace(
        computation,
        ppo_primary_transition_inventory_sha256=_digest("tampered-transitions"),
        semantic_receipt_sha256="0" * 64,
    )
    tampered = replace(
        tampered,
        semantic_receipt_sha256=semantic_sha256(tampered.semantic_unsigned()),
    )
    with pytest.raises(ValueError, match="transition inventory differs"):
        tampered.validate()


def test_full_cold_replay_is_nonmaterializing(tmp_path: Path, real_vertical) -> None:
    before = tuple(tmp_path.rglob("*"))
    replayed = tuple(_fold_report(row.computation) for row in real_vertical)
    after = tuple(tmp_path.rglob("*"))
    assert before == after == ()
    assert tuple(row.fold_index for row in replayed) == (0, 1, 2, 3)


def test_real_v5_vertical_executes_without_economic_mocks(real_vertical) -> None:
    assert tuple(row.fold_index for row in real_vertical) == (0, 1, 2, 3)
    assert all(row.model_updated for row in real_vertical)
    assert len({row.checkpoint_receipt for row in real_vertical}) == 4
    assert all(
        len(row.computation.ppo_primary_transitions) == _SESSION_COUNT
        for row in real_vertical
    )
    assert all(
        sum(
            transition.economic_step.strategy_execution.gross_traded_notional
            for transition in row.computation.ppo_primary_transitions
        )
        > 0.0
        for row in real_vertical
    )
    assert all(
        sum(
            transition.economic_step.strategy_execution.total_transaction_cost
            for transition in row.computation.ppo_primary_transitions
        )
        > 0.0
        for row in real_vertical
    )
    assert all(
        sum(
            transition.economic_step.strategy_execution.total_transaction_cost
            for transition in row.computation._fixed_control_transitions
        )
        > 0.0
        for row in real_vertical
    )
    assert tuple(
        stage.value for stage in MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1
    ) == (MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1)
