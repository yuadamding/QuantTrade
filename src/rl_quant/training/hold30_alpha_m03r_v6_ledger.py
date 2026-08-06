"""Authoritative v5-execution to v6 persistence-ledger adapter.

M03R v6 reuses the qualified v5 execution decomposition while its production
driver remains launch-blocked.  This adapter freezes which executed sale
causes are economically discretionary for the v6 soft-persistence objective.
Policy-requested benchmark de-risking and post-request projection sales are
included with learned-hazard sales, so a policy cannot route a young exit
through projection to avoid the penalty.  Pretrade unavailability and
current-book risk repair remain exempt forced causes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rl_quant.execution.hold30_m03r_projection_v5 import M03RExecutionResult
from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_AGE_LEDGER_BIN_COUNT
from rl_quant.training.hold30_alpha_m03r_v6 import M03RV6ExitNotionalByAge

M03R_V6_PERSISTENCE_LEDGER_ADAPTER_SCHEMA = (
    "rl-quant.hold30.m03r-v6-persistence-ledger-adapter-v1"
)


class M03RV6LedgerAdapterError(ValueError):
    """Execution causes cannot be mapped losslessly into the v6 ledger."""


def _require_sale_age_tensor(
    name: str,
    value: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[-1] != M03R_AGE_LEDGER_BIN_COUNT
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
    ):
        raise M03RV6LedgerAdapterError(
            f"{name} must be finite nonnegative [asset,61] sale-age notional"
        )
    if reference is not None and (
        value.shape != reference.shape
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise M03RV6LedgerAdapterError(
            f"{name} must align exactly with the execution total sale-age tensor"
        )


def _require_sell_vector_matches_age_tensor(
    name: str,
    sell_notional: torch.Tensor,
    sale_age_notional: torch.Tensor,
    *,
    tolerance: float,
) -> None:
    if (
        not isinstance(sell_notional, torch.Tensor)
        or sell_notional.ndim != 1
        or sell_notional.shape[0] != sale_age_notional.shape[0]
        or sell_notional.dtype != sale_age_notional.dtype
        or sell_notional.device != sale_age_notional.device
        or not bool(torch.isfinite(sell_notional).all())
        or bool((sell_notional < 0.0).any())
        or not bool(
            torch.allclose(
                sale_age_notional.sum(dim=-1),
                sell_notional,
                atol=tolerance,
                rtol=tolerance,
            )
        )
    ):
        raise M03RV6LedgerAdapterError(
            f"{name} sell notional does not match its sale-age ledger"
        )


@dataclass(frozen=True, slots=True)
class M03RV6PersistenceLedgerAdapterResult:
    """Cause-typed v6 inventory plus auditable discretionary subcauses."""

    exits: M03RV6ExitNotionalByAge
    learned_hazard: torch.Tensor
    policy_benchmark_derisk: torch.Tensor
    policy_projection: torch.Tensor
    executed_total: torch.Tensor


def adapt_m03r_v5_execution_to_v6_persistence_ledger(
    execution: M03RExecutionResult,
    *,
    tolerance: float = 1.0e-8,
) -> M03RV6PersistenceLedgerAdapterResult:
    """Map one authoritative execution result into the v6 objective inventory."""

    if not isinstance(execution, M03RExecutionResult):
        raise M03RV6LedgerAdapterError(
            "execution must be a typed M03R v5 execution result"
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 < float(tolerance) <= 1.0e-6
    ):
        raise M03RV6LedgerAdapterError("tolerance must lie in (0,1e-6]")
    checked_tolerance = float(tolerance)

    total = execution.executed_total_sale_age_notional
    _require_sale_age_tensor("executed_total_sale_age_notional", total)
    causes = (
        (
            "unavailable",
            execution.executed_unavailable_sale_age_notional,
            execution.executed_unavailable_sell_notional,
        ),
        (
            "risk_repair",
            execution.executed_risk_repair_sale_age_notional,
            execution.executed_risk_repair_sell_notional,
        ),
        (
            "learned_hazard",
            execution.executed_learned_hazard_sale_age_notional,
            execution.executed_learned_hazard_sell_notional,
        ),
        (
            "policy_benchmark_derisk",
            execution.executed_benchmark_derisk_sale_age_notional,
            execution.executed_benchmark_derisk_sell_notional,
        ),
        (
            "policy_projection",
            execution.executed_projection_sale_age_notional,
            execution.executed_projection_sell_notional,
        ),
    )
    for name, sale_age, sell in causes:
        _require_sale_age_tensor(name, sale_age, reference=total)
        _require_sell_vector_matches_age_tensor(
            name,
            sell,
            sale_age,
            tolerance=checked_tolerance,
        )

    partition = sum((sale_age for _, sale_age, _ in causes), torch.zeros_like(total))
    if not bool(
        torch.allclose(
            partition,
            total,
            atol=checked_tolerance,
            rtol=checked_tolerance,
        )
    ):
        raise M03RV6LedgerAdapterError(
            "cause-specific sale-age tensors do not partition executed sales"
        )

    def collapse(value: torch.Tensor) -> torch.Tensor:
        return value.sum(dim=0)

    unavailable = collapse(execution.executed_unavailable_sale_age_notional)
    risk_repair = collapse(execution.executed_risk_repair_sale_age_notional)
    learned_hazard = collapse(execution.executed_learned_hazard_sale_age_notional)
    policy_benchmark_derisk = collapse(
        execution.executed_benchmark_derisk_sale_age_notional
    )
    policy_projection = collapse(execution.executed_projection_sale_age_notional)
    discretionary = learned_hazard + policy_benchmark_derisk + policy_projection
    zero = torch.zeros_like(discretionary)
    exits = M03RV6ExitNotionalByAge(
        discretionary_policy=discretionary,
        other_forced=zero,
        unavailable=unavailable,
        risk_repair=risk_repair,
        corporate_action=zero,
        terminal=zero,
        valid_decision_session_count=1,
    )
    return M03RV6PersistenceLedgerAdapterResult(
        exits=exits,
        learned_hazard=learned_hazard,
        policy_benchmark_derisk=policy_benchmark_derisk,
        policy_projection=policy_projection,
        executed_total=collapse(total),
    )


def adapt_m03r_v5_execution_sequence_to_v6_persistence_ledger(
    executions: Sequence[M03RExecutionResult],
    *,
    tolerance: float = 1.0e-8,
) -> M03RV6PersistenceLedgerAdapterResult:
    """Aggregate an exact nonempty sequence of valid scored decisions.

    The returned cause inventory owns the denominator used by the v6
    persistence objective. Callers cannot supply a different decision count
    alongside the same aggregated age ledger.
    """

    rows = tuple(executions)
    if not rows:
        raise M03RV6LedgerAdapterError(
            "a persistence ledger requires at least one valid scored decision"
        )
    adapted = tuple(
        adapt_m03r_v5_execution_to_v6_persistence_ledger(
            execution,
            tolerance=tolerance,
        )
        for execution in rows
    )

    reference = adapted[0].exits.discretionary_policy
    for row in adapted[1:]:
        candidate = row.exits.discretionary_policy
        if (
            candidate.shape != reference.shape
            or candidate.dtype != reference.dtype
            or candidate.device != reference.device
        ):
            raise M03RV6LedgerAdapterError(
                "all scored decisions must use one aligned age-ledger dtype and device"
            )

    def summed_exit_field(name: str) -> torch.Tensor:
        return torch.stack(
            [getattr(row.exits, name) for row in adapted],
            dim=0,
        ).sum(dim=0)

    exits = M03RV6ExitNotionalByAge(
        discretionary_policy=summed_exit_field("discretionary_policy"),
        other_forced=summed_exit_field("other_forced"),
        unavailable=summed_exit_field("unavailable"),
        risk_repair=summed_exit_field("risk_repair"),
        corporate_action=summed_exit_field("corporate_action"),
        terminal=summed_exit_field("terminal"),
        valid_decision_session_count=len(adapted),
    )

    def summed_result_field(name: str) -> torch.Tensor:
        return torch.stack(
            [getattr(row, name) for row in adapted],
            dim=0,
        ).sum(dim=0)

    return M03RV6PersistenceLedgerAdapterResult(
        exits=exits,
        learned_hazard=summed_result_field("learned_hazard"),
        policy_benchmark_derisk=summed_result_field("policy_benchmark_derisk"),
        policy_projection=summed_result_field("policy_projection"),
        executed_total=summed_result_field("executed_total"),
    )


__all__ = [
    "M03R_V6_PERSISTENCE_LEDGER_ADAPTER_SCHEMA",
    "M03RV6LedgerAdapterError",
    "M03RV6PersistenceLedgerAdapterResult",
    "adapt_m03r_v5_execution_sequence_to_v6_persistence_ledger",
    "adapt_m03r_v5_execution_to_v6_persistence_ledger",
]
