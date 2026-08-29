"""Chronological compiler-to-fill-to-book adaptive profit trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    MassiveAdaptiveForecastCalibrationV1,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
)
from rl_quant.evaluation.massive_adaptive_compiler_input_authority_v1 import (
    build_massive_adaptive_compiler_input_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_economic_event_transition_v1 import (
    build_massive_adaptive_economic_event_transition_v1,
)
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    build_massive_adaptive_economic_book_v1,
    build_massive_adaptive_holding_v1,
)
from rl_quant.evaluation.massive_adaptive_execution_result_v1 import (
    execute_massive_adaptive_order_intent_v1,
)
from rl_quant.execution.massive_adaptive_order_intent_v1 import (
    build_massive_adaptive_order_intent_v1,
    build_massive_adaptive_target_order_intent_v1,
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
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)

MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SCHEMA = "rl-quant.massive-adaptive-profit-trace-v1"
MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_DATASET = "massive-adaptive-profit-trace-v1"
MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SCHEMA,
        "payload": "canonical-json-trace-v1",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "chronology": "decision-close-next-session-morning-fill-next-close-mark",
        "state": "continuous-cash-and-shares",
        "benchmark": "same-fill-equal-weight-action-support",
        "cost": "declared-one-way-basis-points",
        "caller_returns": False,
        "target_access": False,
        "reporting": False,
        "outer": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveProfitTraceV1Error(ValueError):
    """The adaptive chronological economic trace does not reconcile."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitTraceRowV1:
    decision_session_date: str
    fill_session_date: str
    forecast_row_receipt_sha256: str
    compiler_input_authority_receipt_sha256: str
    compiler_decision_receipt_sha256: str
    decision_security_ids: tuple[str, ...]
    decision_target_weights: tuple[float, ...]
    decision_target_receipt_sha256: str
    order_intent_receipt_sha256: str
    execution_receipt_sha256: str
    benchmark_order_intent_receipt_sha256: str
    benchmark_execution_receipt_sha256: str
    pretrade_book_receipt_sha256: str
    posttrade_book_receipt_sha256: str
    benchmark_pretrade_book_receipt_sha256: str
    benchmark_posttrade_book_receipt_sha256: str
    pretrade_equity: float
    posttrade_equity: float
    benchmark_pretrade_equity: float
    benchmark_posttrade_equity: float
    gross_return: float
    net_return: float
    benchmark_net_return: float
    active_log_return: float
    turnover: float
    transaction_cost: float
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        values = (
            self.pretrade_equity,
            self.posttrade_equity,
            self.benchmark_pretrade_equity,
            self.benchmark_posttrade_equity,
            self.gross_return,
            self.net_return,
            self.benchmark_net_return,
            self.active_log_return,
            self.turnover,
            self.transaction_cost,
        )
        expected_net = self.posttrade_equity / self.pretrade_equity - 1.0
        expected_benchmark = (
            self.benchmark_posttrade_equity / self.benchmark_pretrade_equity - 1.0
        )
        expected_active = math.log1p(expected_net) - math.log1p(expected_benchmark)
        if (
            not self.decision_session_date
            or self.fill_session_date <= self.decision_session_date
            or any(not math.isfinite(value) for value in values)
            or self.decision_security_ids
            != tuple(sorted(set(self.decision_security_ids)))
            or len(self.decision_security_ids) != len(self.decision_target_weights)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.decision_target_weights
            )
            or sum(self.decision_target_weights) > 1.0 + 1.0e-8
            or self.decision_target_receipt_sha256
            != semantic_sha256(
                (self.decision_security_ids, self.decision_target_weights)
            )
            or min(
                self.pretrade_equity,
                self.posttrade_equity,
                self.benchmark_pretrade_equity,
                self.benchmark_posttrade_equity,
                self.turnover,
                self.transaction_cost,
            )
            < 0.0
            or abs(self.net_return - expected_net) > 1.0e-12
            or abs(self.benchmark_net_return - expected_benchmark) > 1.0e-12
            or abs(self.active_log_return - expected_active) > 1.0e-12
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveProfitTraceV1Error("adaptive profit row differs")


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitTraceV1:
    rows: tuple[MassiveAdaptiveProfitTraceRowV1, ...]
    forecast_archive_receipt_sha256: str
    calibration_receipt_sha256: str
    inference_plan_receipt_sha256: str
    fill_source_receipt_sha256: str
    daily_input_receipt_sha256: str
    identity_authority_receipt_sha256: str
    economic_event_authority_inventory_sha256: str
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
    specification_sha256: str = MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SCHEMA

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
            self.schema != MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SCHEMA
            or not self.rows
            or tuple(row.decision_session_date for row in self.rows)
            != tuple(sorted(set(row.decision_session_date for row in self.rows)))
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not isinstance(self.economic_event_transition_qualified, bool)
            or not isinstance(self.frozen_actions_replayed, bool)
            or self.frozen_actions_replayed
            != (self.frozen_decision_trace_receipt_sha256 is not None)
            or not isinstance(self.source_data_qualified, bool)
            or self.source_data_qualified
            and not self.economic_event_transition_qualified
            or not isinstance(self.deterministic_profitability_replayed, bool)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitTraceV1Error("adaptive profit trace differs")
        for index, row in enumerate(self.rows):
            row.validate()
            if index and (
                row.pretrade_book_receipt_sha256
                != self.rows[index - 1].posttrade_book_receipt_sha256
                or row.benchmark_pretrade_book_receipt_sha256
                != self.rows[index - 1].benchmark_posttrade_book_receipt_sha256
            ):
                raise MassiveAdaptiveProfitTraceV1Error(
                    "adaptive book does not carry continuously"
                )
        if (
            abs(self.final_equity - self.rows[-1].posttrade_equity) > 1.0e-12
            or abs(
                self.final_benchmark_equity - self.rows[-1].benchmark_posttrade_equity
            )
            > 1.0e-12
            or abs(
                self.cumulative_active_log_return
                - sum(row.active_log_return for row in self.rows)
            )
            > 1.0e-12
        ):
            raise MassiveAdaptiveProfitTraceV1Error("adaptive final wealth differs")
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())
        if self.loaded_source is not None:
            self.loaded_source.validate()
            if (
                self.loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_DATASET
                or self.loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SOURCE_SCHEMA_SHA256
                or self.loaded_source.receipt.entitlement_receipt_sha256
                != self.source_inventory_sha256
            ):
                raise MassiveAdaptiveProfitTraceV1Error(
                    "adaptive profit trace source transaction differs"
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


def build_massive_adaptive_profit_trace_v1(
    *,
    forecast_archive: MassiveAdaptiveForecastArchiveV2,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    inference_plan: MassiveAdaptiveInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None = None,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV1 | None = None,
    initial_capital: float,
    transaction_cost_basis_points: float = 20.0,
    maximum_fill_participation: float = 0.02,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitTraceV1:
    """Run the exact target-free validation chronology from source roots."""

    forecast_archive.validate()
    calibration.validate()
    inference_plan.validate()
    fill_source.validate()
    daily_input_authority.validate()
    identity_authority.validate()
    if economic_event_archive is not None:
        economic_event_archive.validate()
    if frozen_decision_trace is not None:
        frozen_decision_trace.validate()
        if (
            frozen_decision_trace.frozen_actions_replayed
            or frozen_decision_trace.inference_plan_receipt_sha256
            != inference_plan.semantic_receipt_sha256
            or frozen_decision_trace.fill_source_receipt_sha256
            != fill_source.semantic_receipt_sha256
            or frozen_decision_trace.daily_input_receipt_sha256
            != daily_input_authority.semantic_receipt_sha256
            or frozen_decision_trace.identity_authority_receipt_sha256
            != identity_authority.receipt_sha256
            or frozen_decision_trace.initial_capital != initial_capital
            or transaction_cost_basis_points
            == frozen_decision_trace.transaction_cost_basis_points
        ):
            raise MassiveAdaptiveProfitTraceV1Error(
                "frozen action trace has incompatible economic roots or cost rung"
            )
    if (
        forecast_archive.runtime_rows is None
        or not forecast_archive.runtime_forecasts_replayed
    ):
        raise MassiveAdaptiveProfitTraceV1Error("forecast runtime is not replayed")
    if not math.isfinite(initial_capital) or initial_capital <= 0.0:
        raise MassiveAdaptiveProfitTraceV1Error("initial capital is invalid")
    root_by_date = {row.decision_session_date: row for row in decision_roots}
    context_by_date = {row.decision_session_date: row for row in context_origins}
    forecast_by_date = {
        row.decision_session_date: row for row in forecast_archive.runtime_rows
    }
    frozen_by_date = (
        {}
        if frozen_decision_trace is None
        else {row.decision_session_date: row for row in frozen_decision_trace.rows}
    )
    if tuple(forecast_by_date) != tuple(
        row.decision_session_date for row in inference_plan.rows
    ):
        raise MassiveAdaptiveProfitTraceV1Error("forecast and inference dates differ")
    config = compiler_config or MassiveAdaptivePortfolioCompilerConfigV1()
    config.validate()
    initialization = semantic_sha256(
        {
            "initial_capital": initial_capital,
            "forecast": forecast_archive.semantic_receipt_sha256,
            "inference": inference_plan.semantic_receipt_sha256,
        }
    )
    first_date = inference_plan.rows[0].decision_session_date
    first_forecast = forecast_by_date[first_date]
    initial_ids = tuple(
        security_id
        for security_id, valid in zip(
            first_forecast.security_ids, first_forecast.valid, strict=True
        )
        if bool(valid)
    )
    if not initial_ids:
        raise MassiveAdaptiveProfitTraceV1Error(
            "the feasible benchmark has no first-decision support"
        )
    initial_marks, initial_mark_receipts = _decision_marks(
        session_date=first_date,
        security_ids=initial_ids,
        daily=daily_input_authority,
    )
    if set(initial_marks) != set(initial_ids):
        raise MassiveAdaptiveProfitTraceV1Error(
            "the feasible benchmark lacks first-decision close marks"
        )
    masters = {row.security_id: row for row in identity_authority.security_master}
    initial_holdings = tuple(
        build_massive_adaptive_holding_v1(
            security_id=security_id,
            issuer_id=masters[security_id].issuer_id,
            shares=(initial_capital / len(initial_ids)) / initial_marks[security_id],
            mark=initial_marks[security_id],
            identity_receipt_sha256=masters[security_id].identity_source_receipt_sha256,
            mark_receipt_sha256=initial_mark_receipts[security_id],
        )
        for security_id in initial_ids
    )
    book = build_massive_adaptive_economic_book_v1(
        decision_session_date=first_date,
        cash=0.0,
        holdings=initial_holdings,
        prior_high_water_mark=initial_capital,
        source_state_receipt_sha256=initialization,
    )
    benchmark_book = build_massive_adaptive_economic_book_v1(
        decision_session_date=first_date,
        cash=0.0,
        holdings=initial_holdings,
        prior_high_water_mark=initial_capital,
        source_state_receipt_sha256=semantic_sha256((initialization, "benchmark")),
    )
    rows: list[MassiveAdaptiveProfitTraceRowV1] = []
    all_compiler_qualified = True
    all_event_qualified = economic_event_archive is not None
    for plan_row in inference_plan.rows:
        forecast_row = forecast_by_date[plan_row.decision_session_date]
        root = root_by_date[plan_row.decision_session_date]
        context = context_by_date[plan_row.decision_session_date]
        authority = build_massive_adaptive_compiler_input_authority_v1(
            forecast_archive=forecast_archive,
            forecast_row=forecast_row,
            calibration=calibration,
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
                frozen_row = frozen_by_date[plan_row.decision_session_date]
            except KeyError as exc:
                raise MassiveAdaptiveProfitTraceV1Error(
                    "frozen action trace does not cover the profit chronology"
                ) from exc
            decision_security_ids = frozen_row.decision_security_ids
            decision_target_weights = frozen_row.decision_target_weights
            compiler_decision_receipt = frozen_row.compiler_decision_receipt_sha256
        else:
            decision_security_ids = decision.security_ids
            decision_target_weights = decision.target_weights
            compiler_decision_receipt = decision.semantic_receipt_sha256
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
            raise MassiveAdaptiveProfitTraceV1Error(
                "a requested or held security lacks a decision-close mark"
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
                target_receipt_sha256=compiler_decision_receipt,
                decision_marks=marks,
                decision_mark_receipts=mark_receipts,
            )
        )
        transition = None
        if economic_event_archive is not None:
            transition = build_massive_adaptive_economic_event_transition_v1(
                prior_session_date=plan_row.decision_session_date,
                fill_session_date=plan_row.next_session_date,
                provider_archive=economic_event_archive,
                daily_input_authority=daily_input_authority,
                identity_authority=identity_authority,
            )
        execution = execute_massive_adaptive_order_intent_v1(
            order_intent=intent,
            book=book,
            fill_source=fill_source,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            economic_event_transition=transition,
            transaction_cost_basis_points=transaction_cost_basis_points,
            maximum_fill_participation=maximum_fill_participation,
        )
        # C1 is initialized equal weight once and then buys-and-drifts.  Its
        # current economic weights are therefore the no-trade target here.
        benchmark_weights = benchmark_book.weights(
            authority.runtime_inputs.security_ids
        )
        benchmark_required = tuple(
            security_id
            for security_id, target, current in zip(
                authority.runtime_inputs.security_ids,
                benchmark_weights,
                benchmark_book.weights(authority.runtime_inputs.security_ids),
                strict=True,
            )
            if target > 1.0e-12 or current > 1.0e-12
        )
        benchmark_marks, benchmark_mark_receipts = _decision_marks(
            session_date=plan_row.decision_session_date,
            security_ids=benchmark_required,
            daily=daily_input_authority,
        )
        benchmark_target_receipt = semantic_sha256(
            (
                plan_row.receipt_sha256,
                authority.runtime_inputs.security_ids,
                benchmark_weights,
            )
        )
        benchmark_intent = build_massive_adaptive_target_order_intent_v1(
            decision_session_date=plan_row.decision_session_date,
            scheduled_fill_session_date=plan_row.next_session_date,
            book=benchmark_book,
            security_ids=authority.runtime_inputs.security_ids,
            target_weights=benchmark_weights,
            target_receipt_sha256=benchmark_target_receipt,
            decision_marks=benchmark_marks,
            decision_mark_receipts=benchmark_mark_receipts,
        )
        benchmark_execution = execute_massive_adaptive_order_intent_v1(
            order_intent=benchmark_intent,
            book=benchmark_book,
            fill_source=fill_source,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            economic_event_transition=transition,
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
            "compiler_decision_receipt_sha256": compiler_decision_receipt,
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
            "turnover": execution.gross_traded_notional / book.marked_equity / 2.0,
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
    economic_event_transition_qualified = all_event_qualified
    source_qualified = bool(
        isinstance(forecast_archive, MassiveAdaptiveForecastArchiveV2)
        and isinstance(calibration, MassiveAdaptiveForecastCalibrationV1)
        and isinstance(inference_plan, MassiveAdaptiveInferencePlanV1)
        and all(isinstance(root, MassiveAdaptiveDecisionRootV1) for root in decision_roots)
        and all(
            isinstance(context, MassiveAdaptiveContextOriginAuthorityV1)
            for context in context_origins
        )
        and isinstance(fill_source, MassiveAdaptiveFillSourceV1)
        and isinstance(
            daily_input_authority, MassiveProfitabilityDailyInputAuthorityV1
        )
        and isinstance(identity_authority, PITSecurityUniverseAuthority)
        and isinstance(
            economic_event_archive, MassiveProviderEconomicArchiveAuthorityV6
        )
        and (
            frozen_decision_trace is None
            or frozen_decision_trace.source_data_qualified
        )
        and all_compiler_qualified
        and fill_source.source_data_qualified
        and daily_input_authority.daily_input_data_qualified
        and economic_event_transition_qualified
    )
    source_inventory = semantic_sha256(
        (
            forecast_archive.semantic_receipt_sha256,
            calibration.semantic_receipt_sha256,
            inference_plan.semantic_receipt_sha256,
            fill_source.semantic_receipt_sha256,
            daily_input_authority.semantic_receipt_sha256,
            identity_authority.receipt_sha256,
            tuple(root.semantic_receipt_sha256 for root in decision_roots),
            tuple(context.semantic_receipt_sha256 for context in context_origins),
            None
            if economic_event_archive is None
            else economic_event_archive.receipt_sha256,
            None
            if frozen_decision_trace is None
            else frozen_decision_trace.semantic_receipt_sha256,
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SCHEMA,
        "rows": tuple(rows),
        "forecast_archive_receipt_sha256": forecast_archive.semantic_receipt_sha256,
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "inference_plan_receipt_sha256": inference_plan.semantic_receipt_sha256,
        "fill_source_receipt_sha256": fill_source.semantic_receipt_sha256,
        "daily_input_receipt_sha256": daily_input_authority.semantic_receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "economic_event_authority_inventory_sha256": semantic_sha256(
            ()
            if economic_event_archive is None
            else (economic_event_archive.receipt_sha256,)
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
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_inventory_sha256": source_inventory,
        "economic_event_transition_qualified": (economic_event_transition_qualified),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveProfitTraceV1(
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


def _artifact_id(value: str) -> str:
    if not value or any(
        not (character.isalnum() or character in "-_") for character in value
    ):
        raise MassiveAdaptiveProfitTraceV1Error(
            "adaptive profit trace artifact ID is not path safe"
        )
    return value


def _payload(trace: MassiveAdaptiveProfitTraceV1) -> dict[str, object]:
    payload = trace.semantic_unsigned()
    payload["rows"] = tuple(asdict(row) for row in trace.rows)
    return {
        **payload,
        "semantic_receipt_sha256": trace.semantic_receipt_sha256,
    }


def parse_massive_adaptive_profit_trace_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveProfitTraceV1:
    """Load committed metrics while withholding replay authority."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveProfitTraceV1Error(
            "adaptive profit trace payload is not canonical JSON"
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
    result = MassiveAdaptiveProfitTraceV1(
        **payload,
        rows=rows,
        deterministic_profitability_replayed=False,
        loaded_source=loaded_source,
    )
    result.validate()
    return result


def authorize_massive_adaptive_profit_trace_v1(
    *,
    root: str | Path,
    trace: MassiveAdaptiveProfitTraceV1,
    forecast_archive: MassiveAdaptiveForecastArchiveV2,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    inference_plan: MassiveAdaptiveInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None = None,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV1 | None = None,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitTraceV1:
    """Rebuild every decision, fill, and book before restoring replay status."""

    if trace.loaded_source is None:
        raise MassiveAdaptiveProfitTraceV1Error(
            "adaptive profit trace is not attached to a committed source object"
        )
    parsed = parse_massive_adaptive_profit_trace_v1(
        root=root, loaded_source=trace.loaded_source
    )
    rebuilt = build_massive_adaptive_profit_trace_v1(
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
        raise MassiveAdaptiveProfitTraceV1Error(
            "adaptive profit trace does not replay from committed roots"
        )
    result = replace(
        rebuilt,
        loaded_source=parsed.loaded_source,
        deterministic_profitability_replayed=True,
    )
    result.validate()
    return result


def materialize_massive_adaptive_profit_trace_v1(
    *,
    root: str | Path,
    artifact_id: str,
    forecast_archive: MassiveAdaptiveForecastArchiveV2,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    inference_plan: MassiveAdaptiveInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None = None,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV1 | None = None,
    initial_capital: float,
    committed_at_ms: int,
    transaction_cost_basis_points: float = 20.0,
    maximum_fill_participation: float = 0.02,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitTraceV1:
    """Publish once, generically reload, then reexecute the full trace."""

    identifier = _artifact_id(artifact_id)
    built = build_massive_adaptive_profit_trace_v1(
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
    relative = f"massive-adaptive/profit-trace-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(built))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_PROFIT_TRACE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=built.source_inventory_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-PROFIT-TRACE-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_profit_trace_v1(root=root, loaded_source=loaded)
    return authorize_massive_adaptive_profit_trace_v1(
        root=root,
        trace=generic,
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
    "MassiveAdaptiveProfitTraceRowV1",
    "MassiveAdaptiveProfitTraceV1",
    "MassiveAdaptiveProfitTraceV1Error",
    "authorize_massive_adaptive_profit_trace_v1",
    "build_massive_adaptive_profit_trace_v1",
    "materialize_massive_adaptive_profit_trace_v1",
    "parse_massive_adaptive_profit_trace_v1",
]
