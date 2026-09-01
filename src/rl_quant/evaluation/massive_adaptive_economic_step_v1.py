"""One authoritative, duration-free adaptive portfolio economic transition.

Both the deterministic baseline and an adaptive RL controller settle through
this module.  The policy may alter only the bounded compiler preferences; the
fill, event, cash/share, benchmark, and reward accounting paths are shared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_benchmark_authority_v1 import (
    MassiveAdaptiveBenchmarkAuthorityV1,
    build_massive_adaptive_benchmark_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_compiler_input_authority_v2 import (
    MassiveAdaptiveCompilerInputAuthorityV2,
    build_massive_adaptive_compiler_input_authority_v2,
)
from rl_quant.evaluation.massive_adaptive_economic_event_transition_v2 import (
    MassiveAdaptiveEconomicEventTransitionV2,
    build_massive_adaptive_economic_event_transition_v2,
)
from rl_quant.evaluation.massive_adaptive_execution_result_v1 import (
    MassiveAdaptiveExecutionResultV1,
    execute_massive_adaptive_order_intent_v1,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastRowV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_runtime_protocol_v1 import (
    MassiveAdaptiveForecastRuntimeProtocol,
    MassiveAdaptiveInferenceRowRuntimeProtocol,
)
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
)
from rl_quant.execution.massive_adaptive_order_intent_v1 import (
    build_massive_adaptive_order_intent_v1,
    build_massive_adaptive_target_order_intent_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerInputsV1,
    MassiveAdaptivePortfolioDecisionV1,
)
from rl_quant.execution.massive_adaptive_rl_compiler_control_v1 import (
    MassiveAdaptiveRLCompilerControlV1,
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

MASSIVE_ADAPTIVE_PREPARED_STEP_V1_SCHEMA = (
    "rl-quant.massive-adaptive-prepared-economic-step-v1"
)
MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SCHEMA = "rl-quant.massive-adaptive-economic-step-v1"
MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SPEC_SHA256 = semantic_sha256(
    {
        "chronology": "decision-close-next-session-morning-fill-next-close-mark",
        "books": ("strategy", "neutral", "benchmark"),
        "reward": "realized-log-wealth-only",
        "policy_constraints": "compiler-hard-envelope",
        "cost_stress": "primary-targets-replayed-without-policy-or-compiler-rerun",
        "target_access": False,
        "future_fill_access": False,
    }
)


class MassiveAdaptiveEconomicStepV1Error(ValueError):
    """The prepared roots, decisions, execution, or wealth do not reconcile."""


def _digest(value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveEconomicStepV1Error("economic-step digest is invalid")


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePreparedStepV1:
    decision_session_date: str
    fill_session_date: str
    strategy_compiler_input_authority: MassiveAdaptiveCompilerInputAuthorityV2
    neutral_compiler_input_authority: MassiveAdaptiveCompilerInputAuthorityV2
    benchmark_authority: MassiveAdaptiveBenchmarkAuthorityV1
    strategy_pretrade_book: MassiveAdaptiveEconomicBookV1
    neutral_pretrade_book: MassiveAdaptiveEconomicBookV1
    benchmark_pretrade_book: MassiveAdaptiveEconomicBookV1
    forecast_row_receipt_sha256: str
    inference_row_receipt_sha256: str
    source_inventory_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PREPARED_STEP_V1_SCHEMA

    @property
    def strategy_compiler_inputs(self) -> MassiveAdaptivePortfolioCompilerInputsV1:
        value = self.strategy_compiler_input_authority.runtime_inputs
        if value is None:
            raise MassiveAdaptiveEconomicStepV1Error(
                "strategy compiler inputs are not replayed"
            )
        return value

    @property
    def neutral_compiler_inputs(self) -> MassiveAdaptivePortfolioCompilerInputsV1:
        value = self.neutral_compiler_input_authority.runtime_inputs
        if value is None:
            raise MassiveAdaptiveEconomicStepV1Error(
                "neutral compiler inputs are not replayed"
            )
        return value

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for value in (
            self.strategy_compiler_input_authority,
            self.neutral_compiler_input_authority,
            self.benchmark_authority,
            self.strategy_pretrade_book,
            self.neutral_pretrade_book,
            self.benchmark_pretrade_book,
        ):
            value.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_PREPARED_STEP_V1_SCHEMA
            or not self.decision_session_date
            or self.fill_session_date <= self.decision_session_date
            or self.strategy_pretrade_book.decision_session_date
            != self.decision_session_date
            or self.neutral_pretrade_book.decision_session_date
            != self.decision_session_date
            or self.benchmark_pretrade_book.decision_session_date
            != self.decision_session_date
            or self.strategy_compiler_inputs.decision_id != self.decision_session_date
            or self.neutral_compiler_inputs.decision_id != self.decision_session_date
            or self.strategy_compiler_input_authority.economic_book_receipt_sha256
            != self.strategy_pretrade_book.semantic_receipt_sha256
            or self.neutral_compiler_input_authority.economic_book_receipt_sha256
            != self.neutral_pretrade_book.semantic_receipt_sha256
            or self.benchmark_authority.benchmark_book_receipt_sha256
            != self.benchmark_pretrade_book.semantic_receipt_sha256
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveEconomicStepV1Error(
                "prepared adaptive economic step differs"
            )
        for receipt in (
            self.forecast_row_receipt_sha256,
            self.inference_row_receipt_sha256,
            self.source_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest(receipt)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def prepare_massive_adaptive_economic_step_v1(
    *,
    forecast_archive: MassiveAdaptiveForecastRuntimeProtocol,
    forecast_row: MassiveAdaptiveForecastRowV2,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    inference_row: MassiveAdaptiveInferenceRowRuntimeProtocol,
    decision_root: MassiveAdaptiveDecisionRootV1,
    context_origin: MassiveAdaptiveContextOriginAuthorityV1,
    strategy_book: MassiveAdaptiveEconomicBookV1,
    neutral_book: MassiveAdaptiveEconomicBookV1,
    benchmark_book: MassiveAdaptiveEconomicBookV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveAdaptivePreparedStepV1:
    """Build causal compiler inputs for strategy and neutral shadow books."""

    for value in (strategy_book, neutral_book, benchmark_book):
        value.validate()
        if value.decision_session_date != inference_row.decision_session_date:
            raise MassiveAdaptiveEconomicStepV1Error(
                "economic book and inference chronology differ"
            )
    axis = tuple(
        sorted(
            set(forecast_row.security_ids)
            | set(strategy_book.shares_by_security())
            | set(neutral_book.shares_by_security())
            | set(benchmark_book.shares_by_security())
        )
    )
    archive_qualified = bool(
        getattr(forecast_archive, "development_forecast_authorized", False)
        or getattr(forecast_archive, "outer_forecast_authorized", False)
    )
    benchmark = build_massive_adaptive_benchmark_authority_v1(
        decision_session_date=inference_row.decision_session_date,
        security_ids=axis,
        forecast_security_ids=forecast_row.security_ids,
        forecast_valid=tuple(bool(value) for value in forecast_row.valid),
        forecast_row_receipt_sha256=forecast_row.receipt_sha256,
        benchmark_book=benchmark_book,
        source_data_qualified=archive_qualified,
    )
    strategy_authority = build_massive_adaptive_compiler_input_authority_v2(
        forecast_archive=forecast_archive,
        forecast_row=forecast_row,
        calibration=calibration,
        benchmark_authority=benchmark,
        decision_root=decision_root,
        context_origin=context_origin,
        inference_row=inference_row,
        book=strategy_book,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
    )
    neutral_authority = build_massive_adaptive_compiler_input_authority_v2(
        forecast_archive=forecast_archive,
        forecast_row=forecast_row,
        calibration=calibration,
        benchmark_authority=benchmark,
        decision_root=decision_root,
        context_origin=context_origin,
        inference_row=inference_row,
        book=neutral_book,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
    )
    source_inventory = semantic_sha256(
        (
            forecast_archive.semantic_receipt_sha256,
            forecast_row.receipt_sha256,
            calibration.semantic_receipt_sha256,
            inference_row.receipt_sha256,
            decision_root.semantic_receipt_sha256,
            context_origin.semantic_receipt_sha256,
            strategy_authority.semantic_receipt_sha256,
            neutral_authority.semantic_receipt_sha256,
            benchmark.semantic_receipt_sha256,
            daily_input_authority.semantic_receipt_sha256,
            identity_authority.receipt_sha256,
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_PREPARED_STEP_V1_SCHEMA,
        "decision_session_date": inference_row.decision_session_date,
        "fill_session_date": inference_row.next_session_date,
        "strategy_compiler_input_authority": strategy_authority,
        "neutral_compiler_input_authority": neutral_authority,
        "benchmark_authority": benchmark,
        "strategy_pretrade_book": strategy_book,
        "neutral_pretrade_book": neutral_book,
        "benchmark_pretrade_book": benchmark_book,
        "forecast_row_receipt_sha256": forecast_row.receipt_sha256,
        "inference_row_receipt_sha256": inference_row.receipt_sha256,
        "source_inventory_sha256": source_inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptivePreparedStepV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _decision_marks(
    *,
    session_date: str,
    security_ids: tuple[str, ...],
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
    if set(marks) != set(security_ids):
        raise MassiveAdaptiveEconomicStepV1Error(
            "economic-step security lacks a decision-close mark"
        )
    return marks, receipts


def _execute_decision(
    *,
    prepared: MassiveAdaptivePreparedStepV1,
    book: MassiveAdaptiveEconomicBookV1,
    decision: MassiveAdaptivePortfolioDecisionV1,
    fill_source: MassiveAdaptiveFillSourceV1,
    transition: MassiveAdaptiveEconomicEventTransitionV2 | None,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    transaction_cost_basis_points: float,
    maximum_fill_participation: float,
) -> MassiveAdaptiveExecutionResultV1:
    required = tuple(
        security_id
        for security_id, target, current in zip(
            decision.security_ids,
            decision.target_weights,
            book.weights(decision.security_ids),
            strict=True,
        )
        if target > 1.0e-12 or current > 1.0e-12
    )
    marks, receipts = _decision_marks(
        session_date=prepared.decision_session_date,
        security_ids=required,
        daily=daily_input_authority,
    )
    intent = build_massive_adaptive_order_intent_v1(
        book=book,
        decision=decision,
        scheduled_fill_session_date=prepared.fill_session_date,
        decision_marks=marks,
        decision_mark_receipts=receipts,
    )
    return execute_massive_adaptive_order_intent_v1(
        order_intent=intent,
        book=book,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_transition=transition,  # type: ignore[arg-type]
        transaction_cost_basis_points=transaction_cost_basis_points,
        maximum_fill_participation=maximum_fill_participation,
    )


def _execute_frozen_target(
    *,
    prepared: MassiveAdaptivePreparedStepV1,
    book: MassiveAdaptiveEconomicBookV1,
    decision: MassiveAdaptivePortfolioDecisionV1,
    fill_source: MassiveAdaptiveFillSourceV1,
    transition: MassiveAdaptiveEconomicEventTransitionV2 | None,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    transaction_cost_basis_points: float,
    maximum_fill_participation: float,
) -> MassiveAdaptiveExecutionResultV1:
    """Replay a primary target against the current cost-rung book.

    The primary compiler decision is provenance for the frozen target only.  A
    stress book naturally has a different input receipt after prior costs, so
    it must not pretend that the original compiler consumed the stress book.
    """

    required = tuple(
        security_id
        for security_id, target, current in zip(
            decision.security_ids,
            decision.target_weights,
            book.weights(decision.security_ids),
            strict=True,
        )
        if target > 1.0e-12 or current > 1.0e-12
    )
    marks, receipts = _decision_marks(
        session_date=prepared.decision_session_date,
        security_ids=required,
        daily=daily_input_authority,
    )
    target_receipt = semantic_sha256((decision.security_ids, decision.target_weights))
    intent = build_massive_adaptive_target_order_intent_v1(
        decision_session_date=prepared.decision_session_date,
        scheduled_fill_session_date=prepared.fill_session_date,
        book=book,
        security_ids=decision.security_ids,
        target_weights=decision.target_weights,
        target_receipt_sha256=target_receipt,
        decision_marks=marks,
        decision_mark_receipts=receipts,
    )
    return execute_massive_adaptive_order_intent_v1(
        order_intent=intent,
        book=book,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_transition=transition,  # type: ignore[arg-type]
        transaction_cost_basis_points=transaction_cost_basis_points,
        maximum_fill_participation=maximum_fill_participation,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveEconomicStepV1:
    prepared_step_receipt_sha256: str
    policy_action_receipt_sha256: str | None
    policy_control_receipt_sha256: str | None
    policy_decision_receipt_sha256: str
    neutral_decision_receipt_sha256: str
    strategy_execution: MassiveAdaptiveExecutionResultV1
    neutral_execution: MassiveAdaptiveExecutionResultV1
    benchmark_execution: MassiveAdaptiveExecutionResultV1
    strategy_posttrade_book: MassiveAdaptiveEconomicBookV1
    neutral_posttrade_book: MassiveAdaptiveEconomicBookV1
    benchmark_posttrade_book: MassiveAdaptiveEconomicBookV1
    strategy_net_log_return: float
    neutral_net_log_return: float
    benchmark_net_log_return: float
    strategy_active_log_return: float
    neutral_active_log_return: float
    incremental_rl_log_return: float
    optimization_reward_basis_points: float
    neutral_equivalence: bool
    frozen_targets_replayed: bool
    economic_event_transition_receipt_sha256: str | None
    source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for value in (
            self.strategy_execution,
            self.neutral_execution,
            self.benchmark_execution,
            self.strategy_posttrade_book,
            self.neutral_posttrade_book,
            self.benchmark_posttrade_book,
        ):
            value.validate()
        values = (
            self.strategy_net_log_return,
            self.neutral_net_log_return,
            self.benchmark_net_log_return,
            self.strategy_active_log_return,
            self.neutral_active_log_return,
            self.incremental_rl_log_return,
            self.optimization_reward_basis_points,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SCHEMA
            or any(not math.isfinite(value) for value in values)
            or self.strategy_execution.posttrade_book != self.strategy_posttrade_book
            or self.neutral_execution.posttrade_book != self.neutral_posttrade_book
            or self.benchmark_execution.posttrade_book != self.benchmark_posttrade_book
            or abs(
                self.strategy_active_log_return
                - (self.strategy_net_log_return - self.benchmark_net_log_return)
            )
            > 1.0e-12
            or abs(
                self.neutral_active_log_return
                - (self.neutral_net_log_return - self.benchmark_net_log_return)
            )
            > 1.0e-12
            or abs(
                self.incremental_rl_log_return
                - (self.strategy_net_log_return - self.neutral_net_log_return)
            )
            > 1.0e-12
            or abs(
                self.optimization_reward_basis_points
                - 10_000.0 * self.incremental_rl_log_return
            )
            > 1.0e-9
            or not isinstance(self.source_data_qualified, bool)
            or not isinstance(self.frozen_targets_replayed, bool)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveEconomicStepV1Error(
                "settled adaptive economic step differs"
            )
        for receipt in (
            self.prepared_step_receipt_sha256,
            self.policy_action_receipt_sha256,
            self.policy_control_receipt_sha256,
            self.policy_decision_receipt_sha256,
            self.neutral_decision_receipt_sha256,
            self.economic_event_transition_receipt_sha256,
            self.source_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest(
                receipt,
                optional=receipt is None,
            )
        if self.neutral_equivalence and (
            self.policy_decision_receipt_sha256 != self.neutral_decision_receipt_sha256
            or self.strategy_execution != self.neutral_execution
            or self.strategy_posttrade_book != self.neutral_posttrade_book
            or self.incremental_rl_log_return != 0.0
        ):
            raise MassiveAdaptiveEconomicStepV1Error(
                "neutral policy did not reproduce the deterministic economic path"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def settle_massive_adaptive_economic_step_v1(
    *,
    prepared: MassiveAdaptivePreparedStepV1,
    policy_decision: MassiveAdaptivePortfolioDecisionV1,
    neutral_decision: MassiveAdaptivePortfolioDecisionV1,
    fill_source: MassiveAdaptiveFillSourceV1,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6 | None,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    transaction_cost_basis_points: float,
    maximum_fill_participation: float,
    policy_action_receipt_sha256: str | None = None,
    policy_control_receipt_sha256: str | None = None,
    policy_control: MassiveAdaptiveRLCompilerControlV1 | None = None,
    frozen_targets_replayed: bool = False,
) -> MassiveAdaptiveEconomicStepV1:
    """Settle strategy, neutral, and benchmark through one source chronology."""

    prepared.validate()
    policy_decision.validate()
    neutral_decision.validate()
    if not isinstance(frozen_targets_replayed, bool):
        raise MassiveAdaptiveEconomicStepV1Error("frozen-target replay flag is invalid")
    if policy_control is not None:
        policy_control.validate()
        if (
            policy_control.adjusted_input_receipt_sha256
            != policy_decision.input_receipt_sha256
            or policy_control_receipt_sha256 != policy_control.semantic_receipt_sha256
            or (
                not frozen_targets_replayed
                and policy_control.base_input_receipt_sha256
                != prepared.strategy_compiler_inputs.receipt_sha256
            )
        ):
            raise MassiveAdaptiveEconomicStepV1Error(
                "policy control and prepared compiler input differ"
            )
    elif (
        policy_decision.input_receipt_sha256
        != prepared.strategy_compiler_inputs.receipt_sha256
    ):
        raise MassiveAdaptiveEconomicStepV1Error(
            "policy decision did not consume the prepared compiler input"
        )
    if (
        neutral_decision.input_receipt_sha256
        != prepared.neutral_compiler_inputs.receipt_sha256
    ):
        raise MassiveAdaptiveEconomicStepV1Error(
            "neutral decision did not consume the prepared compiler input"
        )
    transition = (
        None
        if economic_event_archive is None
        else build_massive_adaptive_economic_event_transition_v2(
            prior_session_date=prepared.decision_session_date,
            fill_session_date=prepared.fill_session_date,
            provider_archive=economic_event_archive,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
        )
    )
    strategy_execution = (
        _execute_frozen_target(
            prepared=prepared,
            book=prepared.strategy_pretrade_book,
            decision=policy_decision,
            fill_source=fill_source,
            transition=transition,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            transaction_cost_basis_points=transaction_cost_basis_points,
            maximum_fill_participation=maximum_fill_participation,
        )
        if frozen_targets_replayed
        else _execute_decision(
            prepared=prepared,
            book=prepared.strategy_pretrade_book,
            decision=policy_decision,
            fill_source=fill_source,
            transition=transition,
            daily_input_authority=daily_input_authority,
            identity_authority=identity_authority,
            transaction_cost_basis_points=transaction_cost_basis_points,
            maximum_fill_participation=maximum_fill_participation,
        )
    )
    neutral_execution = _execute_decision(
        prepared=prepared,
        book=prepared.neutral_pretrade_book,
        decision=neutral_decision,
        fill_source=fill_source,
        transition=transition,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        transaction_cost_basis_points=transaction_cost_basis_points,
        maximum_fill_participation=maximum_fill_participation,
    )
    benchmark_axis = prepared.benchmark_authority.security_ids
    benchmark_required = tuple(
        security_id
        for security_id, target, current in zip(
            benchmark_axis,
            prepared.benchmark_authority.target_weights,
            prepared.benchmark_pretrade_book.weights(benchmark_axis),
            strict=True,
        )
        if target > 1.0e-12 or current > 1.0e-12
    )
    marks, receipts = _decision_marks(
        session_date=prepared.decision_session_date,
        security_ids=benchmark_required,
        daily=daily_input_authority,
    )
    benchmark_intent = build_massive_adaptive_target_order_intent_v1(
        decision_session_date=prepared.decision_session_date,
        scheduled_fill_session_date=prepared.fill_session_date,
        book=prepared.benchmark_pretrade_book,
        security_ids=benchmark_axis,
        target_weights=prepared.benchmark_authority.target_weights,
        target_receipt_sha256=prepared.benchmark_authority.semantic_receipt_sha256,
        decision_marks=marks,
        decision_mark_receipts=receipts,
    )
    benchmark_execution = execute_massive_adaptive_order_intent_v1(
        order_intent=benchmark_intent,
        book=prepared.benchmark_pretrade_book,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_transition=transition,  # type: ignore[arg-type]
        transaction_cost_basis_points=transaction_cost_basis_points,
        maximum_fill_participation=maximum_fill_participation,
    )
    strategy_log = math.log(
        strategy_execution.posttrade_book.marked_equity
        / prepared.strategy_pretrade_book.marked_equity
    )
    neutral_log = math.log(
        neutral_execution.posttrade_book.marked_equity
        / prepared.neutral_pretrade_book.marked_equity
    )
    benchmark_log = math.log(
        benchmark_execution.posttrade_book.marked_equity
        / prepared.benchmark_pretrade_book.marked_equity
    )
    incremental = strategy_log - neutral_log
    neutral_equivalence = bool(
        not frozen_targets_replayed
        and policy_decision.semantic_receipt_sha256
        == neutral_decision.semantic_receipt_sha256
        and prepared.strategy_pretrade_book == prepared.neutral_pretrade_book
    )
    source_inventory = semantic_sha256(
        (
            prepared.source_inventory_sha256,
            fill_source.semantic_receipt_sha256,
            daily_input_authority.semantic_receipt_sha256,
            identity_authority.receipt_sha256,
            None if transition is None else transition.semantic_receipt_sha256,
            strategy_execution.semantic_receipt_sha256,
            neutral_execution.semantic_receipt_sha256,
            benchmark_execution.semantic_receipt_sha256,
            frozen_targets_replayed,
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SCHEMA,
        "prepared_step_receipt_sha256": prepared.semantic_receipt_sha256,
        "policy_action_receipt_sha256": policy_action_receipt_sha256,
        "policy_control_receipt_sha256": policy_control_receipt_sha256,
        "policy_decision_receipt_sha256": policy_decision.semantic_receipt_sha256,
        "neutral_decision_receipt_sha256": neutral_decision.semantic_receipt_sha256,
        "strategy_execution": strategy_execution,
        "neutral_execution": neutral_execution,
        "benchmark_execution": benchmark_execution,
        "strategy_posttrade_book": strategy_execution.posttrade_book,
        "neutral_posttrade_book": neutral_execution.posttrade_book,
        "benchmark_posttrade_book": benchmark_execution.posttrade_book,
        "strategy_net_log_return": strategy_log,
        "neutral_net_log_return": neutral_log,
        "benchmark_net_log_return": benchmark_log,
        "strategy_active_log_return": strategy_log - benchmark_log,
        "neutral_active_log_return": neutral_log - benchmark_log,
        "incremental_rl_log_return": incremental,
        "optimization_reward_basis_points": 10_000.0 * incremental,
        "neutral_equivalence": neutral_equivalence,
        "frozen_targets_replayed": frozen_targets_replayed,
        "economic_event_transition_receipt_sha256": None
        if transition is None
        else transition.semantic_receipt_sha256,
        "source_inventory_sha256": source_inventory,
        "source_data_qualified": bool(
            prepared.strategy_compiler_input_authority.development_compiler_authorized
            and prepared.neutral_compiler_input_authority.development_compiler_authorized
            and prepared.benchmark_authority.source_data_qualified
            and fill_source.source_data_qualified
            and daily_input_authority.daily_input_data_qualified
            and economic_event_archive is not None
            and strategy_execution.economic_event_transition_qualified
            and neutral_execution.economic_event_transition_qualified
            and benchmark_execution.economic_event_transition_qualified
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_ECONOMIC_STEP_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveEconomicStepV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveEconomicStepV1",
    "MassiveAdaptiveEconomicStepV1Error",
    "MassiveAdaptivePreparedStepV1",
    "prepare_massive_adaptive_economic_step_v1",
    "settle_massive_adaptive_economic_step_v1",
]
