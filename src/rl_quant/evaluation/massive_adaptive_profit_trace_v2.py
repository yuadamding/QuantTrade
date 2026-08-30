"""Corrected cash-funded, benchmark-consistent adaptive profit trace V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_benchmark_authority_v1 import (
    build_massive_adaptive_benchmark_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_compiler_input_authority_v2 import (
    build_massive_adaptive_compiler_input_authority_v2,
)
from rl_quant.evaluation.massive_adaptive_economic_event_transition_v2 import (
    build_massive_adaptive_economic_event_transition_v2,
)
from rl_quant.evaluation.massive_adaptive_execution_result_v1 import (
    execute_massive_adaptive_order_intent_v1,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
)
from rl_quant.evaluation.massive_adaptive_initial_book_authority_v1 import (
    build_massive_adaptive_initial_book_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v1 import (
    MassiveAdaptiveProfitTraceRowV1,
)
from rl_quant.execution.massive_adaptive_order_intent_v1 import (
    build_massive_adaptive_order_intent_v1,
    build_massive_adaptive_target_order_intent_v1,
)
from rl_quant.evaluation.massive_adaptive_economic_step_v1 import (
    prepare_massive_adaptive_economic_step_v1,
    settle_massive_adaptive_economic_step_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
    compile_massive_adaptive_portfolio_v1,
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
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
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

MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SCHEMA = "rl-quant.massive-adaptive-profit-trace-v2"
MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_DATASET = "massive-adaptive-profit-trace-v2"
MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SCHEMA,
        "payload": "canonical-json-trace-v2",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "chronology": "cash-close-next-session-morning-fill-next-close-mark",
        "initial_book": "all-cash-authority-v1",
        "benchmark": "same-buy-and-drift-authority-for-compiler-and-return",
        "events": "dual-fill-start-and-close-snapshots-v2",
        "turnover": "max-buy-sell-notional-over-pretrade-equity",
        "state": "continuous-cash-and-shares",
        "caller_returns": False,
        "target_access": False,
        "reporting": False,
        "roles": ("inner_validation", "outer_test"),
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveProfitTraceV2Error(ValueError):
    """The corrected adaptive economic trace does not reconcile."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitTraceV2:
    rows: tuple[MassiveAdaptiveProfitTraceRowV1, ...]
    evaluation_role: str
    fold_index: int
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    calibration_receipt_sha256: str
    inference_plan_receipt_sha256: str
    initial_book_authority_receipt_sha256: str
    benchmark_authority_receipts: tuple[str, ...]
    benchmark_authority_inventory_sha256: str
    fill_source_receipt_sha256: str
    daily_input_receipt_sha256: str
    identity_authority_receipt_sha256: str
    economic_event_authority_inventory_sha256: str
    economic_event_transition_inventory_sha256: str
    frozen_decision_trace_receipt_sha256: str | None
    frozen_actions_replayed: bool
    compiler_config_receipt_sha256: str
    initial_capital: float
    transaction_cost_basis_points: float
    maximum_fill_participation: float
    final_equity: float
    final_benchmark_equity: float
    cumulative_active_log_return: float
    row_inventory_sha256: str
    source_inventory_sha256: str
    economic_event_transition_qualified: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    deterministic_profitability_replayed: bool
    loaded_source: LoadedMassiveSourceObject | None = None
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "deterministic_profitability_replayed",
                "loaded_source",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SCHEMA
            or not self.rows
            or self.evaluation_role not in {"inner_validation", "outer_test"}
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or tuple(row.decision_session_date for row in self.rows)
            != tuple(sorted(set(row.decision_session_date for row in self.rows)))
            or len(self.benchmark_authority_receipts) != len(self.rows)
            or self.benchmark_authority_inventory_sha256
            != semantic_sha256(self.benchmark_authority_receipts)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.frozen_actions_replayed
            != (self.frozen_decision_trace_receipt_sha256 is not None)
            or not isinstance(self.economic_event_transition_qualified, bool)
            or self.source_data_qualified and not self.economic_event_transition_qualified
            or not isinstance(self.deterministic_profitability_replayed, bool)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitTraceV2Error(
                "adaptive profit trace v2 identity differs"
            )
        for index, row in enumerate(self.rows):
            row.validate()
            if index and (
                row.pretrade_book_receipt_sha256
                != self.rows[index - 1].posttrade_book_receipt_sha256
                or row.benchmark_pretrade_book_receipt_sha256
                != self.rows[index - 1].benchmark_posttrade_book_receipt_sha256
            ):
                raise MassiveAdaptiveProfitTraceV2Error(
                    "adaptive V2 book does not carry continuously"
                )
        if (
            abs(self.final_equity - self.rows[-1].posttrade_equity) > 1.0e-12
            or abs(self.final_benchmark_equity - self.rows[-1].benchmark_posttrade_equity)
            > 1.0e-12
            or abs(
                self.cumulative_active_log_return
                - sum(row.active_log_return for row in self.rows)
            )
            > 1.0e-12
        ):
            raise MassiveAdaptiveProfitTraceV2Error("adaptive V2 final wealth differs")
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())
        if self.loaded_source is not None:
            self.loaded_source.validate()
            if (
                self.loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_DATASET
                or self.loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SOURCE_SCHEMA_SHA256
                or self.loaded_source.receipt.entitlement_receipt_sha256
                != self.source_inventory_sha256
            ):
                raise MassiveAdaptiveProfitTraceV2Error(
                    "adaptive V2 trace source transaction differs"
                )


def _decision_marks(
    *,
    session_date: str,
    security_ids: Sequence[str],
    daily: MassiveProfitabilityDailyInputAuthorityV1,
) -> tuple[dict[str, float], dict[str, str]]:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    marks: dict[str, float] = {}
    receipts: dict[str, str] = {}
    for security_id in security_ids:
        row = daily.row(session_date=session_date, security_id=security_id)
        if row.bars_valid[close_index] and row.bars_values[close_index] > 0.0:
            marks[security_id] = float(row.bars_values[close_index])
            receipts[security_id] = row.receipt_sha256
    return marks, receipts


def _lineage(archive: object) -> tuple[int, str, str, str, bool]:
    if isinstance(archive, MassiveAdaptiveForecastArchiveV2):
        return (
            archive.fold_index,
            archive.checkpoint_receipt_sha256,
            archive.model_state_receipt_sha256,
            archive.training_window_plan_receipt_sha256,
            archive.development_forecast_authorized,
        )
    if isinstance(archive, MassiveAdaptiveOuterForecastArchiveV1):
        return (
            archive.fold_index,
            archive.selected_checkpoint_receipt_sha256,
            archive.model_state_receipt_sha256,
            archive.training_window_plan_receipt_sha256,
            archive.outer_forecast_authorized,
        )
    # Structural fixtures may exercise mechanics, but can never qualify source data.
    checkpoint = getattr(archive, "checkpoint_receipt_sha256", None)
    if checkpoint is None:
        checkpoint = getattr(archive, "selected_checkpoint_receipt_sha256", None)
    return (
        int(getattr(archive, "fold_index")),
        str(checkpoint),
        str(getattr(archive, "model_state_receipt_sha256")),
        str(getattr(archive, "training_window_plan_receipt_sha256")),
        False,
    )


def full_portfolio_one_way_turnover_v2(execution: object, equity: float) -> float:
    """Include the cash leg by taking the larger of buys and sells."""

    buy = sum(
        row.executed_notional
        for row in execution.rows
        if row.filled_shares > 0.0
    )
    sell = sum(
        row.executed_notional
        for row in execution.rows
        if row.filled_shares < 0.0
    )
    return max(buy, sell) / equity


def build_massive_adaptive_profit_trace_v2(
    *,
    forecast_archive: MassiveAdaptiveForecastArchiveV2 | MassiveAdaptiveOuterForecastArchiveV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    inference_plan: MassiveAdaptiveInferencePlanV1 | MassiveAdaptiveOuterInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None = None,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV2 | None = None,
    initial_capital: float,
    transaction_cost_basis_points: float = 20.0,
    maximum_fill_participation: float = 0.02,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitTraceV2:
    """Execute the corrected target-free validation or outer chronology."""

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
    role = (
        "outer_test"
        if isinstance(inference_plan, MassiveAdaptiveOuterInferencePlanV1)
        else "inner_validation"
    )
    if role == "outer_test" and not isinstance(
        forecast_archive, MassiveAdaptiveOuterForecastArchiveV1
    ):
        raise MassiveAdaptiveProfitTraceV2Error("outer plan requires outer forecast")
    if role == "inner_validation" and isinstance(
        forecast_archive, MassiveAdaptiveOuterForecastArchiveV1
    ):
        raise MassiveAdaptiveProfitTraceV2Error(
            "inner-validation plan cannot consume outer forecast"
        )
    if not math.isfinite(initial_capital) or initial_capital <= 0.0:
        raise MassiveAdaptiveProfitTraceV2Error("initial capital is invalid")
    if forecast_archive.runtime_rows is None or not forecast_archive.runtime_forecasts_replayed:
        raise MassiveAdaptiveProfitTraceV2Error("forecast runtime is not replayed")
    fold, checkpoint, model_state, training_window, archive_qualified = _lineage(
        forecast_archive
    )
    if (
        calibration.fold_index != fold
        or calibration.checkpoint_receipt_sha256 != checkpoint
        or calibration.model_state_receipt_sha256 != model_state
        or calibration.training_window_plan_receipt_sha256 != training_window
    ):
        raise MassiveAdaptiveProfitTraceV2Error(
            "profit trace calibration and forecast lineage differ"
        )
    if getattr(inference_plan, "fold_index", fold) != fold:
        raise MassiveAdaptiveProfitTraceV2Error(
            "profit trace inference and forecast folds differ"
        )
    if frozen_decision_trace is not None:
        frozen_decision_trace.validate()
        if (
            frozen_decision_trace.frozen_actions_replayed
            or frozen_decision_trace.fold_index != fold
            or frozen_decision_trace.evaluation_role != role
            or frozen_decision_trace.inference_plan_receipt_sha256
            != inference_plan.semantic_receipt_sha256
            or frozen_decision_trace.initial_capital != initial_capital
            or transaction_cost_basis_points
            == frozen_decision_trace.transaction_cost_basis_points
        ):
            raise MassiveAdaptiveProfitTraceV2Error(
                "frozen V2 action trace has incompatible roots or cost rung"
            )
    roots = {row.decision_session_date: row for row in decision_roots}
    contexts = {row.decision_session_date: row for row in context_origins}
    forecasts = {
        row.decision_session_date: row for row in forecast_archive.runtime_rows
    }
    frozen = (
        {}
        if frozen_decision_trace is None
        else {row.decision_session_date: row for row in frozen_decision_trace.rows}
    )
    dates = tuple(row.decision_session_date for row in inference_plan.rows)
    if tuple(forecasts) != dates:
        raise MassiveAdaptiveProfitTraceV2Error("forecast and inference dates differ")
    config = compiler_config or MassiveAdaptivePortfolioCompilerConfigV1()
    config.validate()
    initial = build_massive_adaptive_initial_book_authority_v1(
        decision_session_date=dates[0],
        initial_capital=initial_capital,
        forecast_archive_receipt_sha256=forecast_archive.semantic_receipt_sha256,
        inference_plan_receipt_sha256=inference_plan.semantic_receipt_sha256,
        source_data_qualified=bool(archive_qualified),
    )
    book = initial.strategy_book
    neutral_book = initial.neutral_book
    benchmark_book = initial.benchmark_book
    rows: list[MassiveAdaptiveProfitTraceRowV1] = []
    benchmark_receipts: list[str] = []
    transition_receipts: list[str] = []
    all_compiler_qualified = True
    all_benchmark_qualified = True
    all_event_qualified = economic_event_archive is not None
    for plan_row in inference_plan.rows:
        forecast_row = forecasts[plan_row.decision_session_date]
        root = roots[plan_row.decision_session_date]
        context = contexts[plan_row.decision_session_date]
        if frozen_decision_trace is None:
            prepared = prepare_massive_adaptive_economic_step_v1(
                forecast_archive=forecast_archive,
                forecast_row=forecast_row,
                calibration=calibration,
                inference_row=plan_row,
                decision_root=root,
                context_origin=context,
                strategy_book=book,
                neutral_book=neutral_book,
                benchmark_book=benchmark_book,
                daily_input_authority=daily_input_authority,
                identity_authority=identity_authority,
            )
            benchmark = prepared.benchmark_authority
            authority = prepared.strategy_compiler_input_authority
            neutral_authority = prepared.neutral_compiler_input_authority
            benchmark_receipts.append(benchmark.semantic_receipt_sha256)
            all_benchmark_qualified &= benchmark.source_data_qualified
            all_compiler_qualified &= bool(
                authority.development_compiler_authorized
                and neutral_authority.development_compiler_authorized
            )
            decision = compile_massive_adaptive_portfolio_v1(
                prepared.strategy_compiler_inputs,
                config=config,
            )
            neutral_decision = compile_massive_adaptive_portfolio_v1(
                prepared.neutral_compiler_inputs,
                config=config,
            )
            step = settle_massive_adaptive_economic_step_v1(
                prepared=prepared,
                policy_decision=decision,
                neutral_decision=neutral_decision,
                fill_source=fill_source,
                economic_event_archive=economic_event_archive,
                daily_input_authority=daily_input_authority,
                identity_authority=identity_authority,
                transaction_cost_basis_points=transaction_cost_basis_points,
                maximum_fill_participation=maximum_fill_participation,
            )
            if step.economic_event_transition_receipt_sha256 is not None:
                transition_receipts.append(
                    step.economic_event_transition_receipt_sha256
                )
            execution = step.strategy_execution
            benchmark_execution = step.benchmark_execution
            net_return = math.expm1(step.strategy_net_log_return)
            benchmark_return = math.expm1(step.benchmark_net_log_return)
            gross_return = (
                execution.posttrade_book.marked_equity
                + execution.total_transaction_cost
            ) / book.marked_equity - 1.0
            row_body = {
                "decision_session_date": plan_row.decision_session_date,
                "fill_session_date": plan_row.next_session_date,
                "forecast_row_receipt_sha256": forecast_row.receipt_sha256,
                "compiler_input_authority_receipt_sha256": authority.semantic_receipt_sha256,
                "compiler_decision_receipt_sha256": decision.semantic_receipt_sha256,
                "decision_security_ids": decision.security_ids,
                "decision_target_weights": decision.target_weights,
                "decision_target_receipt_sha256": semantic_sha256(
                    (decision.security_ids, decision.target_weights)
                ),
                "order_intent_receipt_sha256": execution.order_intent_receipt_sha256,
                "execution_receipt_sha256": execution.semantic_receipt_sha256,
                "benchmark_order_intent_receipt_sha256": (
                    benchmark_execution.order_intent_receipt_sha256
                ),
                "benchmark_execution_receipt_sha256": (
                    benchmark_execution.semantic_receipt_sha256
                ),
                "pretrade_book_receipt_sha256": book.semantic_receipt_sha256,
                "posttrade_book_receipt_sha256": (
                    execution.posttrade_book.semantic_receipt_sha256
                ),
                "benchmark_pretrade_book_receipt_sha256": (
                    benchmark_book.semantic_receipt_sha256
                ),
                "benchmark_posttrade_book_receipt_sha256": (
                    benchmark_execution.posttrade_book.semantic_receipt_sha256
                ),
                "pretrade_equity": book.marked_equity,
                "posttrade_equity": execution.posttrade_book.marked_equity,
                "benchmark_pretrade_equity": benchmark_book.marked_equity,
                "benchmark_posttrade_equity": (
                    benchmark_execution.posttrade_book.marked_equity
                ),
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_net_return": benchmark_return,
                "active_log_return": step.strategy_active_log_return,
                "turnover": full_portfolio_one_way_turnover_v2(
                    execution, book.marked_equity
                ),
                "transaction_cost": execution.total_transaction_cost,
            }
            trace_row = MassiveAdaptiveProfitTraceRowV1(
                **row_body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(row_body),
            )
            trace_row.validate()
            rows.append(trace_row)
            all_event_qualified &= bool(
                execution.economic_event_transition_qualified
                and step.neutral_execution.economic_event_transition_qualified
                and benchmark_execution.economic_event_transition_qualified
            )
            book = step.strategy_posttrade_book
            neutral_book = step.neutral_posttrade_book
            benchmark_book = step.benchmark_posttrade_book
            continue
        axis = tuple(
            sorted(
                set(forecast_row.security_ids)
                | set(book.shares_by_security())
                | set(benchmark_book.shares_by_security())
            )
        )
        benchmark = build_massive_adaptive_benchmark_authority_v1(
            decision_session_date=plan_row.decision_session_date,
            security_ids=axis,
            forecast_security_ids=forecast_row.security_ids,
            forecast_valid=forecast_row.valid,
            forecast_row_receipt_sha256=forecast_row.receipt_sha256,
            benchmark_book=benchmark_book,
            source_data_qualified=bool(archive_qualified),
        )
        benchmark_receipts.append(benchmark.semantic_receipt_sha256)
        all_benchmark_qualified &= benchmark.source_data_qualified
        authority = build_massive_adaptive_compiler_input_authority_v2(
            forecast_archive=forecast_archive,
            forecast_row=forecast_row,
            calibration=calibration,
            benchmark_authority=benchmark,
            decision_root=root,
            context_origin=context,
            inference_row=plan_row,
            book=book,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
        )
        assert authority.runtime_inputs is not None
        all_compiler_qualified &= authority.development_compiler_authorized
        decision = (
            compile_massive_adaptive_portfolio_v1(authority.runtime_inputs, config=config)
            if frozen_decision_trace is None
            else None
        )
        if decision is None:
            try:
                frozen_row = frozen[plan_row.decision_session_date]
            except KeyError as exc:
                raise MassiveAdaptiveProfitTraceV2Error(
                    "frozen V2 trace omits a decision"
                ) from exc
            decision_security_ids = frozen_row.decision_security_ids
            decision_target_weights = frozen_row.decision_target_weights
            decision_receipt = frozen_row.compiler_decision_receipt_sha256
        else:
            decision_security_ids = decision.security_ids
            decision_target_weights = decision.target_weights
            decision_receipt = decision.semantic_receipt_sha256
        required = tuple(
            security_id
            for security_id, target, current in zip(
                decision_security_ids,
                decision_target_weights,
                book.weights(decision_security_ids),
                strict=True,
            )
            if target > 1.0e-12 or current > 1.0e-12
        )
        marks, mark_receipts = _decision_marks(
            session_date=plan_row.decision_session_date,
            security_ids=required,
            daily=daily_input_authority,
        )
        if set(marks) != set(required):
            raise MassiveAdaptiveProfitTraceV2Error(
                "requested or held security lacks a decision-close mark"
            )
        intent = (
            build_massive_adaptive_order_intent_v1(
                book=book,
                decision=decision,
                scheduled_fill_session_date=plan_row.next_session_date,
                decision_marks=marks,
                decision_mark_receipts=mark_receipts,
            )
            if decision is not None
            else build_massive_adaptive_target_order_intent_v1(
                decision_session_date=plan_row.decision_session_date,
                scheduled_fill_session_date=plan_row.next_session_date,
                book=book,
                security_ids=decision_security_ids,
                target_weights=decision_target_weights,
                target_receipt_sha256=decision_receipt,
                decision_marks=marks,
                decision_mark_receipts=mark_receipts,
            )
        )
        transition = None
        if economic_event_archive is not None:
            transition = build_massive_adaptive_economic_event_transition_v2(
                prior_session_date=plan_row.decision_session_date,
                fill_session_date=plan_row.next_session_date,
                provider_archive=economic_event_archive,
                daily_input_authority=daily_input_authority,
                identity_authority=identity_authority,
            )
            transition_receipts.append(transition.semantic_receipt_sha256)
        execution = execute_massive_adaptive_order_intent_v1(
            order_intent=intent,
            book=book,
            fill_source=fill_source,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            economic_event_transition=transition,  # type: ignore[arg-type]
            transaction_cost_basis_points=transaction_cost_basis_points,
            maximum_fill_participation=maximum_fill_participation,
        )
        benchmark_required = tuple(
            security_id
            for security_id, target, current in zip(
                benchmark.security_ids,
                benchmark.target_weights,
                benchmark_book.weights(benchmark.security_ids),
                strict=True,
            )
            if target > 1.0e-12 or current > 1.0e-12
        )
        benchmark_marks, benchmark_mark_receipts = _decision_marks(
            session_date=plan_row.decision_session_date,
            security_ids=benchmark_required,
            daily=daily_input_authority,
        )
        if set(benchmark_marks) != set(benchmark_required):
            raise MassiveAdaptiveProfitTraceV2Error(
                "benchmark security lacks a decision-close mark"
            )
        benchmark_intent = build_massive_adaptive_target_order_intent_v1(
            decision_session_date=plan_row.decision_session_date,
            scheduled_fill_session_date=plan_row.next_session_date,
            book=benchmark_book,
            security_ids=benchmark.security_ids,
            target_weights=benchmark.target_weights,
            target_receipt_sha256=benchmark.semantic_receipt_sha256,
            decision_marks=benchmark_marks,
            decision_mark_receipts=benchmark_mark_receipts,
        )
        benchmark_execution = execute_massive_adaptive_order_intent_v1(
            order_intent=benchmark_intent,
            book=benchmark_book,
            fill_source=fill_source,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            economic_event_transition=transition,  # type: ignore[arg-type]
            transaction_cost_basis_points=transaction_cost_basis_points,
            maximum_fill_participation=maximum_fill_participation,
        )
        net_return = execution.posttrade_book.marked_equity / book.marked_equity - 1.0
        benchmark_return = (
            benchmark_execution.posttrade_book.marked_equity
            / benchmark_book.marked_equity
            - 1.0
        )
        gross_return = (
            execution.posttrade_book.marked_equity + execution.total_transaction_cost
        ) / book.marked_equity - 1.0
        row_body = {
            "decision_session_date": plan_row.decision_session_date,
            "fill_session_date": plan_row.next_session_date,
            "forecast_row_receipt_sha256": forecast_row.receipt_sha256,
            "compiler_input_authority_receipt_sha256": authority.semantic_receipt_sha256,
            "compiler_decision_receipt_sha256": decision_receipt,
            "decision_security_ids": decision_security_ids,
            "decision_target_weights": decision_target_weights,
            "decision_target_receipt_sha256": semantic_sha256(
                (decision_security_ids, decision_target_weights)
            ),
            "order_intent_receipt_sha256": intent.semantic_receipt_sha256,
            "execution_receipt_sha256": execution.semantic_receipt_sha256,
            "benchmark_order_intent_receipt_sha256": benchmark_intent.semantic_receipt_sha256,
            "benchmark_execution_receipt_sha256": benchmark_execution.semantic_receipt_sha256,
            "pretrade_book_receipt_sha256": book.semantic_receipt_sha256,
            "posttrade_book_receipt_sha256": execution.posttrade_book.semantic_receipt_sha256,
            "benchmark_pretrade_book_receipt_sha256": benchmark_book.semantic_receipt_sha256,
            "benchmark_posttrade_book_receipt_sha256": benchmark_execution.posttrade_book.semantic_receipt_sha256,
            "pretrade_equity": book.marked_equity,
            "posttrade_equity": execution.posttrade_book.marked_equity,
            "benchmark_pretrade_equity": benchmark_book.marked_equity,
            "benchmark_posttrade_equity": benchmark_execution.posttrade_book.marked_equity,
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_net_return": benchmark_return,
            "active_log_return": math.log1p(net_return) - math.log1p(benchmark_return),
            "turnover": full_portfolio_one_way_turnover_v2(
                execution, book.marked_equity
            ),
            "transaction_cost": execution.total_transaction_cost,
        }
        trace_row = MassiveAdaptiveProfitTraceRowV1(
            **row_body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(row_body),
        )
        trace_row.validate()
        rows.append(trace_row)
        all_event_qualified &= bool(
            execution.economic_event_transition_qualified
            and benchmark_execution.economic_event_transition_qualified
        )
        book = execution.posttrade_book
        benchmark_book = benchmark_execution.posttrade_book
    source_qualified = bool(
        archive_qualified
        and calibration.development_calibration_authorized
        and initial.source_data_qualified
        and all_benchmark_qualified
        and all_compiler_qualified
        and fill_source.source_data_qualified
        and daily_input_authority.daily_input_data_qualified
        and all_event_qualified
        and all(isinstance(root, MassiveAdaptiveDecisionRootV1) for root in decision_roots)
        and all(
            isinstance(context, MassiveAdaptiveContextOriginAuthorityV1)
            for context in context_origins
        )
        and isinstance(economic_event_archive, MassiveProviderEconomicArchiveAuthorityV6)
        and (frozen_decision_trace is None or frozen_decision_trace.source_data_qualified)
    )
    source_inventory = semantic_sha256(
        (
            forecast_archive.semantic_receipt_sha256,
            calibration.semantic_receipt_sha256,
            inference_plan.semantic_receipt_sha256,
            initial.semantic_receipt_sha256,
            tuple(benchmark_receipts),
            fill_source.semantic_receipt_sha256,
            daily_input_authority.semantic_receipt_sha256,
            identity_authority.receipt_sha256,
            tuple(root.semantic_receipt_sha256 for root in decision_roots),
            tuple(context.semantic_receipt_sha256 for context in context_origins),
            None if economic_event_archive is None else economic_event_archive.receipt_sha256,
            tuple(transition_receipts),
            None if frozen_decision_trace is None else frozen_decision_trace.semantic_receipt_sha256,
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SCHEMA,
        "rows": tuple(rows),
        "evaluation_role": role,
        "fold_index": fold,
        "checkpoint_receipt_sha256": checkpoint,
        "model_state_receipt_sha256": model_state,
        "training_window_plan_receipt_sha256": training_window,
        "forecast_archive_receipt_sha256": forecast_archive.semantic_receipt_sha256,
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "inference_plan_receipt_sha256": inference_plan.semantic_receipt_sha256,
        "initial_book_authority_receipt_sha256": initial.semantic_receipt_sha256,
        "benchmark_authority_receipts": tuple(benchmark_receipts),
        "benchmark_authority_inventory_sha256": semantic_sha256(tuple(benchmark_receipts)),
        "fill_source_receipt_sha256": fill_source.semantic_receipt_sha256,
        "daily_input_receipt_sha256": daily_input_authority.semantic_receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "economic_event_authority_inventory_sha256": semantic_sha256(
            () if economic_event_archive is None else (economic_event_archive.receipt_sha256,)
        ),
        "economic_event_transition_inventory_sha256": semantic_sha256(
            tuple(transition_receipts)
        ),
        "frozen_decision_trace_receipt_sha256": None
        if frozen_decision_trace is None
        else frozen_decision_trace.semantic_receipt_sha256,
        "frozen_actions_replayed": frozen_decision_trace is not None,
        "compiler_config_receipt_sha256": config.receipt_sha256,
        "initial_capital": initial_capital,
        "transaction_cost_basis_points": transaction_cost_basis_points,
        "maximum_fill_participation": maximum_fill_participation,
        "final_equity": book.marked_equity,
        "final_benchmark_equity": benchmark_book.marked_equity,
        "cumulative_active_log_return": sum(row.active_log_return for row in rows),
        "row_inventory_sha256": semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        "source_inventory_sha256": source_inventory,
        "economic_event_transition_qualified": all_event_qualified,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveProfitTraceV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        deterministic_profitability_replayed=True,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _payload(trace: MassiveAdaptiveProfitTraceV2) -> dict[str, object]:
    payload = trace.semantic_unsigned()
    payload["rows"] = tuple(asdict(row) for row in trace.rows)
    return {**payload, "semantic_receipt_sha256": trace.semantic_receipt_sha256}


def parse_massive_adaptive_profit_trace_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveProfitTraceV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveProfitTraceV2Error(
            "adaptive profit trace v2 payload is not canonical JSON"
        )
    rows = tuple(
        MassiveAdaptiveProfitTraceRowV1(
            **{
                **row,
                "decision_security_ids": tuple(row["decision_security_ids"]),
                "decision_target_weights": tuple(row["decision_target_weights"]),
            }
        )
        for row in payload.pop("rows")
    )
    payload["benchmark_authority_receipts"] = tuple(
        payload["benchmark_authority_receipts"]
    )
    result = MassiveAdaptiveProfitTraceV2(
        **payload,
        rows=rows,
        deterministic_profitability_replayed=False,
        loaded_source=loaded_source,
    )
    result.validate()
    return result


def authorize_massive_adaptive_profit_trace_v2(
    *,
    root: str | Path,
    trace: MassiveAdaptiveProfitTraceV2,
    forecast_archive: MassiveAdaptiveForecastArchiveV2 | MassiveAdaptiveOuterForecastArchiveV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    inference_plan: MassiveAdaptiveInferencePlanV1 | MassiveAdaptiveOuterInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None = None,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV2 | None = None,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitTraceV2:
    if trace.loaded_source is None:
        raise MassiveAdaptiveProfitTraceV2Error(
            "adaptive V2 trace is not attached to a committed source"
        )
    parsed = parse_massive_adaptive_profit_trace_v2(
        root=root, loaded_source=trace.loaded_source
    )
    rebuilt = build_massive_adaptive_profit_trace_v2(
        forecast_archive=forecast_archive,
        calibration=calibration,
        inference_plan=inference_plan,
        decision_roots=decision_roots,
        context_origins=context_origins,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_archive=economic_event_archive,
        frozen_decision_trace=frozen_decision_trace,
        initial_capital=parsed.initial_capital,
        transaction_cost_basis_points=parsed.transaction_cost_basis_points,
        maximum_fill_participation=parsed.maximum_fill_participation,
        compiler_config=compiler_config,
    )
    if (
        parsed.semantic_unsigned() != rebuilt.semantic_unsigned()
        or parsed.semantic_receipt_sha256 != rebuilt.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveProfitTraceV2Error(
            "adaptive V2 trace does not replay from committed roots"
        )
    result = replace(
        rebuilt,
        loaded_source=parsed.loaded_source,
        deterministic_profitability_replayed=True,
    )
    result.validate()
    return result


def materialize_massive_adaptive_profit_trace_v2(
    *,
    root: str | Path,
    artifact_id: str,
    forecast_archive: MassiveAdaptiveForecastArchiveV2 | MassiveAdaptiveOuterForecastArchiveV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    inference_plan: MassiveAdaptiveInferencePlanV1 | MassiveAdaptiveOuterInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None = None,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV2 | None = None,
    initial_capital: float,
    committed_at_ms: int,
    transaction_cost_basis_points: float = 20.0,
    maximum_fill_participation: float = 0.02,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitTraceV2:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveProfitTraceV2Error(
            "adaptive V2 trace artifact ID is not path safe"
        )
    built = build_massive_adaptive_profit_trace_v2(
        forecast_archive=forecast_archive,
        calibration=calibration,
        inference_plan=inference_plan,
        decision_roots=decision_roots,
        context_origins=context_origins,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_archive=economic_event_archive,
        frozen_decision_trace=frozen_decision_trace,
        initial_capital=initial_capital,
        transaction_cost_basis_points=transaction_cost_basis_points,
        maximum_fill_participation=maximum_fill_participation,
        compiler_config=compiler_config,
    )
    relative = f"massive-adaptive/profit-trace-v2/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(built))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_PROFIT_TRACE_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=built.source_inventory_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-PROFIT-TRACE-V2-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return authorize_massive_adaptive_profit_trace_v2(
        root=root,
        trace=parse_massive_adaptive_profit_trace_v2(root=root, loaded_source=loaded),
        forecast_archive=forecast_archive,
        calibration=calibration,
        inference_plan=inference_plan,
        decision_roots=decision_roots,
        context_origins=context_origins,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_archive=economic_event_archive,
        frozen_decision_trace=frozen_decision_trace,
        compiler_config=compiler_config,
    )


__all__ = [
    "MassiveAdaptiveProfitTraceV2",
    "MassiveAdaptiveProfitTraceV2Error",
    "authorize_massive_adaptive_profit_trace_v2",
    "build_massive_adaptive_profit_trace_v2",
    "full_portfolio_one_way_turnover_v2",
    "materialize_massive_adaptive_profit_trace_v2",
    "parse_massive_adaptive_profit_trace_v2",
]
