"""Cause-partition regressions for the v6 persistence-ledger adapter."""

from __future__ import annotations

import pytest
import torch

from rl_quant.execution.hold30_m03r_projection_v5 import M03RExecutionResult
from rl_quant.training.hold30_alpha_m03r_v6 import (
    M03RV6TrainingPlan,
    M03RV6TrainingProgress,
    m03r_v6_soft_persistence_objective,
)
from rl_quant.training.hold30_alpha_m03r_v6_ledger import (
    M03RV6LedgerAdapterError,
    adapt_m03r_v5_execution_sequence_to_v6_persistence_ledger,
    adapt_m03r_v5_execution_to_v6_persistence_ledger,
)


def _sale(age: int, notional: float, *, asset: int = 0) -> torch.Tensor:
    value = torch.zeros((2, 61), dtype=torch.float64)
    value[asset, age] = notional
    return value


def _execution() -> M03RExecutionResult:
    result = object.__new__(M03RExecutionResult)
    age_causes = {
        "unavailable": _sale(2, 0.01, asset=0),
        "risk_repair": _sale(3, 0.02, asset=1),
        "learned_hazard": _sale(4, 0.03, asset=0),
        "benchmark_derisk": _sale(5, 0.04, asset=1),
        "projection": _sale(6, 0.05, asset=0),
    }
    for cause, sale_age in age_causes.items():
        object.__setattr__(
            result,
            f"executed_{cause}_sale_age_notional",
            sale_age,
        )
        object.__setattr__(
            result,
            f"executed_{cause}_sell_notional",
            sale_age.sum(dim=-1),
        )
    object.__setattr__(
        result,
        "executed_total_sale_age_notional",
        sum(age_causes.values(), torch.zeros((2, 61), dtype=torch.float64)),
    )
    return result


def test_policy_projection_and_derisk_sales_cannot_bypass_penalty() -> None:
    adapted = adapt_m03r_v5_execution_to_v6_persistence_ledger(_execution())
    expected_discretionary = torch.zeros(61, dtype=torch.float64)
    expected_discretionary[4] = 0.03
    expected_discretionary[5] = 0.04
    expected_discretionary[6] = 0.05
    torch.testing.assert_close(
        adapted.exits.discretionary_policy,
        expected_discretionary,
    )
    assert float(adapted.exits.unavailable.sum()) == pytest.approx(0.01)
    assert float(adapted.exits.risk_repair.sum()) == pytest.approx(0.02)

    loss, _ = m03r_v6_soft_persistence_objective(
        adapted.exits,
        M03RV6TrainingProgress(
            completed_optimizer_steps=100,
            training_plan=M03RV6TrainingPlan(total_optimizer_steps=100),
        ),
    )
    assert float(loss) > 0.0


def test_sequence_adapter_binds_the_objective_denominator_to_scored_rows() -> None:
    single = adapt_m03r_v5_execution_to_v6_persistence_ledger(_execution())
    paired = adapt_m03r_v5_execution_sequence_to_v6_persistence_ledger(
        (_execution(), _execution())
    )
    assert single.exits.valid_decision_session_count == 1
    assert paired.exits.valid_decision_session_count == 2
    torch.testing.assert_close(
        paired.exits.discretionary_policy,
        2.0 * single.exits.discretionary_policy,
    )

    progress = M03RV6TrainingProgress(
        completed_optimizer_steps=100,
        training_plan=M03RV6TrainingPlan(total_optimizer_steps=100),
    )
    single_loss, _ = m03r_v6_soft_persistence_objective(single.exits, progress)
    paired_loss, _ = m03r_v6_soft_persistence_objective(paired.exits, progress)
    assert float(paired_loss) == pytest.approx(float(single_loss))


def test_adapter_rejects_missing_or_overlapping_cause_partition() -> None:
    execution = _execution()
    total = execution.executed_total_sale_age_notional.clone()
    total[0, 6] += 0.01
    object.__setattr__(execution, "executed_total_sale_age_notional", total)
    with pytest.raises(M03RV6LedgerAdapterError, match="do not partition"):
        adapt_m03r_v5_execution_to_v6_persistence_ledger(execution)


def test_adapter_revalidates_sale_vector_against_age_ledger() -> None:
    execution = _execution()
    object.__setattr__(
        execution,
        "executed_projection_sell_notional",
        torch.zeros(2, dtype=torch.float64),
    )
    with pytest.raises(M03RV6LedgerAdapterError, match="does not match"):
        adapt_m03r_v5_execution_to_v6_persistence_ledger(execution)
