"""Soft-persistence checkpoint-selection qualification for immutable M03R v6."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v6 import M03R_CANONICAL_SETTING_ID
from rl_quant.training.hold30_alpha_m03r_v6_selection import (
    M03R_V6_SELECTION_ADAPTER_AVAILABLE,
    M03RV6CandidateEvaluatorIdentity,
    M03RV6CheckpointSelectionContract,
    M03RV6FoldSeed,
    M03RV6SelectionError,
    M03RV6ValidationMetrics,
    order_m03r_v6_metrics_for_qualification,
    select_m03r_v6_checkpoint,
)


def _inventory() -> tuple[M03RV6FoldSeed, ...]:
    return tuple(
        M03RV6FoldSeed(f"fold-{fold:02d}", seed)
        for fold in range(6)
        for seed in range(5)
    )


def _contract(**changes: object) -> M03RV6CheckpointSelectionContract:
    values: dict[str, object] = {
        "setting_id": M03R_CANONICAL_SETTING_ID,
        "expected_fold_seed_inventory": _inventory(),
        "inference_contract_sha256": "1" * 64,
        "common_evaluator_inputs_sha256": "2" * 64,
        "evaluator_implementation_sha256": "3" * 64,
        "require_exact_age_ledger_content_binding": True,
        "maximum_requested_executed_projection_distance": 0.05,
        "maximum_forced_turnover_fraction": 0.10,
    }
    values.update(changes)
    return M03RV6CheckpointSelectionContract(**values)  # type: ignore[arg-type]


def _exit_ages(total: float = 2.0) -> tuple[float, ...]:
    ages = [0.0] * 61
    ages[10] = 0.1 * total
    ages[30] = 0.9 * total
    return tuple(ages)


def _metrics(**changes: object) -> M03RV6ValidationMetrics:
    row = M03RV6ValidationMetrics(
        update=8,
        net_active_return_20bp=0.02,
        net_active_return_40bp=0.01,
        block_bootstrap_lcb95_net_active_return_20bp=0.001,
        annual_tracking_error=0.04,
        active_market_beta=0.02,
        active_beta_equivalence_upper_bound=0.08,
        notional_survival_at_20_sessions=0.60,
        notional_survival_at_30_sessions=0.30,
        restricted_mean_holding_time_through_60_sessions=30.0,
        early_exit_penalty_paid=0.02,
        discretionary_exit_notional_by_age=_exit_ages(),
        fold_censored_notional_fraction=0.20,
        requested_executed_projection_distance=0.01,
        forced_turnover_fraction=0.02,
        information_ratio_20bp=0.50,
        total_portfolio_sharpe_20bp=1.00,
        maximum_drawdown_20bp=0.10,
        mean_daily_one_way_discretionary_turnover=0.03,
        discretionary_turnover_cost_20bp=0.01,
    )
    return replace(row, **changes)  # type: ignore[arg-type]


def _candidate_identity(
    contract: M03RV6CheckpointSelectionContract,
    *,
    update: int,
    policy_character: str,
    receipt_character: str,
    **changes: object,
) -> M03RV6CandidateEvaluatorIdentity:
    values: dict[str, object] = {
        "update": update,
        "common_evaluator_inputs_sha256": (
            contract.common_evaluator_inputs_sha256
        ),
        "candidate_policy_returns_sha256": policy_character * 64,
        "candidate_evaluator_receipt_sha256": receipt_character * 64,
        "inference_contract_sha256": contract.inference_contract_sha256,
        "evaluator_implementation_sha256": (
            contract.evaluator_implementation_sha256
        ),
        "protocol_generation": contract.protocol_generation,
        "design_id": contract.design_id,
        "setting_id": contract.setting_id,
    }
    values.update(changes)
    return M03RV6CandidateEvaluatorIdentity(**values)  # type: ignore[arg-type]


def test_holding_duration_is_telemetry_not_a_hard_eligibility_gate() -> None:
    contract = _contract()
    short = _metrics(
        notional_survival_at_20_sessions=0.05,
        notional_survival_at_30_sessions=0.01,
        restricted_mean_holding_time_through_60_sessions=5.0,
    )
    long = _metrics(
        notional_survival_at_20_sessions=1.0,
        notional_survival_at_30_sessions=0.99,
        restricted_mean_holding_time_through_60_sessions=60.0,
    )
    assert contract.metrics_satisfy_hard_gates(short)
    assert contract.metrics_satisfy_hard_gates(long)
    hard_gates = contract.canonical_payload()["hard_eligibility_gates"]
    assert hard_gates["holding_duration_requirement"] is None
    assert "notional_survival_at_20_sessions" not in hard_gates
    assert "notional_survival_at_30_sessions" not in hard_gates
    assert "restricted_mean_holding_time_through_60_sessions" not in hard_gates


def test_profitable_rmst45_high_right_censor_candidate_remains_eligible() -> None:
    candidate = _metrics(
        net_active_return_20bp=0.03,
        net_active_return_40bp=0.015,
        block_bootstrap_lcb95_net_active_return_20bp=0.005,
        notional_survival_at_20_sessions=0.95,
        notional_survival_at_30_sessions=0.90,
        restricted_mean_holding_time_through_60_sessions=45.0,
        fold_censored_notional_fraction=0.99,
    )
    contract = _contract()
    assert contract.metrics_satisfy_hard_gates(candidate)
    assert order_m03r_v6_metrics_for_qualification(
        (candidate,),
        contract=contract,
    ) == (candidate,)

    payload = contract.canonical_payload()
    hard_gates = payload["hard_eligibility_gates"]
    assert "maximum_fold_censored_notional_fraction" not in hard_gates
    assert hard_gates["exact_age_ledger_bin_count"] == 61
    assert hard_gates["exact_age_ledger_content_binding_required"] is True
    assert "fold_censored_notional_fraction" in payload["bound_non_gating_telemetry"]


def test_zero_discretionary_exits_do_not_fail_hard_eligibility() -> None:
    zero_exit = _metrics(
        discretionary_exit_notional_by_age=tuple(0.0 for _ in range(61)),
        mean_daily_one_way_discretionary_turnover=0.0,
        discretionary_turnover_cost_20bp=0.0,
    )
    assert zero_exit.discretionary_exit_notional == 0.0
    assert _contract().metrics_satisfy_hard_gates(zero_exit)


def test_profitability_lcb_outranks_a_more_thirty_looking_holding_path() -> None:
    economically_stronger = _metrics(
        update=10,
        block_bootstrap_lcb95_net_active_return_20bp=0.010,
        restricted_mean_holding_time_through_60_sessions=18.0,
    )
    thirty_looking = _metrics(
        update=20,
        block_bootstrap_lcb95_net_active_return_20bp=0.005,
        restricted_mean_holding_time_through_60_sessions=30.0,
    )
    ordered = order_m03r_v6_metrics_for_qualification(
        (thirty_looking, economically_stronger),
        contract=_contract(),
    )
    assert ordered[0] is economically_stronger


def test_point_active_return_is_not_ranked_ahead_of_information_ratio() -> None:
    high_point_lower_ir = _metrics(
        update=10,
        net_active_return_20bp=0.20,
        information_ratio_20bp=0.40,
    )
    low_point_higher_ir = _metrics(
        update=20,
        net_active_return_20bp=0.01,
        information_ratio_20bp=0.60,
    )
    ordered = order_m03r_v6_metrics_for_qualification(
        (high_point_lower_ir, low_point_higher_ir),
        contract=_contract(),
    )
    assert ordered[0] is low_point_higher_ir


def test_persistence_breaks_only_otherwise_economic_ties() -> None:
    short = _metrics(
        update=20,
        restricted_mean_holding_time_through_60_sessions=18.0,
    )
    persistent = _metrics(
        update=30,
        restricted_mean_holding_time_through_60_sessions=30.0,
    )
    assert (
        order_m03r_v6_metrics_for_qualification(
            (short, persistent), contract=_contract()
        )[0]
        is persistent
    )

    cheaper_short = replace(short, discretionary_turnover_cost_20bp=0.009)
    assert (
        order_m03r_v6_metrics_for_qualification(
            (cheaper_short, persistent), contract=_contract()
        )[0]
        is cheaper_short
    )


def test_holding_preference_is_one_sided_below_25_sessions() -> None:
    assert (
        _metrics(
            restricted_mean_holding_time_through_60_sessions=45.0
        ).holding_preference_score
        == 0.0
    )
    assert (
        _metrics(
            restricted_mean_holding_time_through_60_sessions=18.0
        ).holding_preference_score
        == -7.0
    )


def test_economic_risk_and_data_quality_gates_remain_hard() -> None:
    contract = _contract()
    assert not contract.metrics_satisfy_hard_gates(_metrics(net_active_return_20bp=0.0))
    assert not contract.metrics_satisfy_hard_gates(
        _metrics(net_active_return_40bp=-1e-9)
    )
    assert not contract.metrics_satisfy_hard_gates(
        _metrics(annual_tracking_error=0.060001)
    )
    assert not contract.metrics_satisfy_hard_gates(
        _metrics(active_beta_equivalence_upper_bound=0.100001)
    )
    assert contract.metrics_satisfy_hard_gates(
        _metrics(fold_censored_notional_fraction=1.0)
    )
    assert not contract.metrics_satisfy_hard_gates(
        _metrics(requested_executed_projection_distance=0.050001)
    )
    assert not contract.metrics_satisfy_hard_gates(
        _metrics(forced_turnover_fraction=0.100001)
    )


def test_validation_payload_content_binds_every_metric_field() -> None:
    baseline = _metrics()
    payload = baseline.canonical_payload()
    dataclass_field_names = {item.name for item in fields(baseline)}
    assert dataclass_field_names.issubset(payload)
    assert payload["discretionary_exit_notional_by_age"] == list(_exit_ages())
    assert payload["discretionary_exit_notional"] == pytest.approx(2.0)

    mutations: dict[str, object] = {
        "update": 9,
        "net_active_return_20bp": 0.021,
        "net_active_return_40bp": 0.011,
        "block_bootstrap_lcb95_net_active_return_20bp": 0.002,
        "annual_tracking_error": 0.041,
        "active_market_beta": 0.021,
        "active_beta_equivalence_upper_bound": 0.081,
        "notional_survival_at_20_sessions": 0.61,
        "notional_survival_at_30_sessions": 0.31,
        "restricted_mean_holding_time_through_60_sessions": 31.0,
        "early_exit_penalty_paid": 0.021,
        "discretionary_exit_notional_by_age": _exit_ages(2.1),
        "fold_censored_notional_fraction": 0.21,
        "requested_executed_projection_distance": 0.011,
        "forced_turnover_fraction": 0.021,
        "information_ratio_20bp": 0.51,
        "total_portfolio_sharpe_20bp": 1.01,
        "maximum_drawdown_20bp": 0.11,
        "mean_daily_one_way_discretionary_turnover": 0.031,
        "discretionary_turnover_cost_20bp": 0.011,
    }
    for name, value in mutations.items():
        assert (
            replace(
                baseline,
                **{name: value},  # type: ignore[arg-type]
            ).receipt_sha256
            != baseline.receipt_sha256
        )


def test_contract_payload_binds_rank_order_and_fails_closed_without_ledger() -> None:
    contract = _contract()
    payload = contract.canonical_payload()
    assert payload["rank_order"][-3:] == [
        "discretionary_turnover_cost_20bp:ascending",
        "holding_preference_score:descending-weak-tiebreak",
        "update:ascending",
    ]
    assert payload["chronological_selection_adapter"]["available"] is False
    assert M03R_V6_SELECTION_ADAPTER_AVAILABLE is False
    with pytest.raises(M03RV6SelectionError, match="exact age-ledger"):
        _contract(require_exact_age_ledger_content_binding=False)
    with pytest.raises(M03RV6SelectionError, match="adapter is unavailable"):
        select_m03r_v6_checkpoint(
            M03R_CANONICAL_SETTING_ID,
            (),
            contract=contract,
        )
    assert (
        replace(
            contract,
            maximum_forced_turnover_fraction=0.11,
        ).receipt_sha256
        != contract.receipt_sha256
    )


def test_multiple_checkpoint_candidates_share_only_contract_common_inputs() -> None:
    contract = _contract()
    first = _candidate_identity(
        contract,
        update=8,
        policy_character="4",
        receipt_character="5",
    )
    second = _candidate_identity(
        contract,
        update=16,
        policy_character="6",
        receipt_character="7",
    )

    first.validate_against(contract)
    second.validate_against(contract)
    assert (
        first.common_evaluator_inputs_sha256
        == second.common_evaluator_inputs_sha256
        == contract.common_evaluator_inputs_sha256
    )
    assert (
        first.candidate_policy_returns_sha256
        != second.candidate_policy_returns_sha256
    )
    assert (
        first.candidate_evaluator_receipt_sha256
        != second.candidate_evaluator_receipt_sha256
    )
    assert first.receipt_sha256 != second.receipt_sha256


def test_checkpoint_candidate_rejects_common_or_candidate_identity_mutation() -> None:
    contract = _contract()
    candidate = _candidate_identity(
        contract,
        update=8,
        policy_character="4",
        receipt_character="5",
    )
    candidate.validate_against(contract)

    wrong_common = replace(candidate, common_evaluator_inputs_sha256="8" * 64)
    with pytest.raises(M03RV6SelectionError, match="common_evaluator_inputs_sha256"):
        wrong_common.validate_against(contract)

    changed_policy = replace(candidate, candidate_policy_returns_sha256="9" * 64)
    changed_receipt = replace(candidate, candidate_evaluator_receipt_sha256="a" * 64)
    assert changed_policy.receipt_sha256 != candidate.receipt_sha256
    assert changed_receipt.receipt_sha256 != candidate.receipt_sha256
