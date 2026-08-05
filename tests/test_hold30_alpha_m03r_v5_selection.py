"""Fail-closed selection and uncertainty-aware gate tests for M03R v5."""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from rl_quant.evaluation.hold30_alpha_m03r_v5 import (
    build_m03r_inference_plan,
    evaluate_m03r_inference,
    m03r_candidate_policy_returns_sha256,
    m03r_common_evaluator_inputs_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_CANONICAL_SETTING_ID,
    M03R_SUPERSEDED_PROTOCOL_GENERATION,
)
from rl_quant.training.hold30_alpha_m03r_v5_selection import (
    M03R_SELECTION_ADAPTER_AVAILABLE,
    M03R_SELECTION_ADAPTER_BLOCKERS,
    M03R_SELECTION_ADAPTER_SCHEMA,
    M03RCheckpointSelectionContract,
    M03RFoldSeed,
    M03RSelectionError,
    M03RValidationMetrics,
    build_m03r_verified_inner_evaluator_receipt,
    select_m03r_checkpoint,
    validate_m03r_chronological_evaluator_evidence,
)


def _inventory() -> tuple[M03RFoldSeed, ...]:
    return tuple(
        M03RFoldSeed(f"fold-{fold:02d}", seed)
        for fold in range(6)
        for seed in range(5)
    )


def _contract(**changes: object) -> M03RCheckpointSelectionContract:
    fields: dict[str, object] = {
        "setting_id": M03R_CANONICAL_SETTING_ID,
        "expected_fold_seed_inventory": _inventory(),
        "inference_contract_sha256": "1" * 64,
        "common_evaluator_inputs_sha256": "2" * 64,
        "evaluator_implementation_sha256": "e" * 64,
        "minimum_notional_survival_at_20_sessions": 0.40,
        "minimum_notional_survival_at_30_sessions": 0.20,
        "minimum_restricted_mean_holding_time_through_60_sessions": 20.0,
        "maximum_restricted_mean_holding_time_through_60_sessions": 40.0,
        "minimum_discretionary_sold_notional": 1.0,
        "maximum_fold_censored_notional_fraction": 0.50,
        "maximum_requested_executed_projection_distance": 0.05,
        "maximum_forced_turnover_fraction": 0.10,
    }
    fields.update(changes)
    return M03RCheckpointSelectionContract(**fields)  # type: ignore[arg-type]


def _metrics(**changes: object) -> M03RValidationMetrics:
    row = M03RValidationMetrics(
        update=8,
        net_active_return_20bp=0.02,
        net_active_return_40bp=0.01,
        block_bootstrap_lcb95_net_active_return_20bp=0.001,
        annual_tracking_error=0.0,
        active_market_beta=0.0,
        active_beta_equivalence_upper_bound=0.09,
        notional_survival_at_20_sessions=0.60,
        notional_survival_at_30_sessions=0.30,
        restricted_mean_holding_time_through_60_sessions=30.0,
        discretionary_sold_notional=2.0,
        fold_censored_notional_fraction=0.20,
        requested_executed_projection_distance=0.01,
        forced_turnover_fraction=0.02,
        information_ratio_20bp=0.5,
        total_portfolio_sharpe_20bp=1.0,
        maximum_drawdown_20bp=0.1,
        turnover_cost_20bp=0.01,
    )
    return replace(row, **changes)


def _chronological_evidence() -> tuple[dict[str, object], dict[str, object]]:
    rng = np.random.default_rng(911)
    shape = (6, 63)
    start = date(2020, 1, 1)
    score_dates = np.asarray(
        [
            (start + timedelta(days=offset)).isoformat()
            for offset in range(shape[0] * shape[1])
        ],
        dtype=object,
    ).reshape(shape)
    fold_ids = np.asarray(
        [[f"fold-{fold:02d}"] * shape[1] for fold in range(shape[0])],
        dtype=object,
    )
    risk_free = np.full(shape, 0.0001)
    market = rng.normal(0.0003, 0.008, size=shape)
    factor_returns = rng.normal(0.0, 0.003, size=(*shape, 1))
    benchmark = risk_free + 0.95 * (market - risk_free) + rng.normal(
        0.0, 0.001, size=shape
    )
    policy = benchmark + 0.00005 + 0.02 * (market - risk_free) + rng.normal(
        0.0, 0.0008, size=shape
    )
    plan = build_m03r_inference_plan(
        factor_names=("SIZE",),
        factor_return_conventions=("daily-simple-long-short-return",),
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256="1" * 64,
        one_sided_alpha=0.05,
    )
    common = m03r_common_evaluator_inputs_sha256(
        score_dates=score_dates,
        fold_ids=fold_ids,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factor_returns,
        plan=plan,
    )
    candidate = m03r_candidate_policy_returns_sha256(
        policy_net_returns=policy,
        common_evaluator_inputs_sha256=common,
        plan=plan,
    )
    arrays: dict[str, object] = {
        "score_dates": score_dates,
        "fold_ids": fold_ids,
        "policy_net_returns": policy,
        "benchmark_net_returns": benchmark,
        "risk_free_returns": risk_free,
        "market_total_returns": market,
        "factor_returns": factor_returns,
        "plan": plan,
    }
    receipt = evaluate_m03r_inference(
        setting_id=M03R_CANONICAL_SETTING_ID,
        common_evaluator_inputs_sha256=common,
        candidate_policy_returns_sha256=candidate,
        **arrays,
    )
    return receipt, arrays


def test_selection_contract_records_adapter_blockers_and_fails_closed() -> None:
    contract = _contract()
    adapter = contract.canonical_payload()["chronological_selection_adapter"]
    assert adapter == {
        "schema": M03R_SELECTION_ADAPTER_SCHEMA,
        "available": False,
        "blockers": list(M03R_SELECTION_ADAPTER_BLOCKERS),
    }
    assert M03R_SELECTION_ADAPTER_AVAILABLE is False
    with pytest.raises(M03RSelectionError, match="adapter is unavailable"):
        select_m03r_checkpoint(
            M03R_CANONICAL_SETTING_ID,
            (),
            contract=contract,
        )


def test_unresolved_numerical_gates_fail_before_adapter_status() -> None:
    unresolved = M03RCheckpointSelectionContract(
        setting_id=M03R_CANONICAL_SETTING_ID,
        expected_fold_seed_inventory=_inventory(),
        inference_contract_sha256="1" * 64,
        common_evaluator_inputs_sha256="2" * 64,
        evaluator_implementation_sha256="e" * 64,
    )
    with pytest.raises(M03RSelectionError, match="remain unresolved"):
        unresolved.require_resolved()


def test_result_gate_uses_beta_equivalence_upper_bound_not_point_beta() -> None:
    contract = _contract()
    assert contract.metrics_satisfy_result_gates(
        _metrics(active_market_beta=0.099, active_beta_equivalence_upper_bound=0.10)
    )
    assert not contract.metrics_satisfy_result_gates(
        _metrics(active_market_beta=0.0, active_beta_equivalence_upper_bound=0.100001)
    )
    with pytest.raises(M03RSelectionError, match="cannot be negative"):
        _metrics(active_beta_equivalence_upper_bound=-1e-6)


def test_result_gate_retains_zero_tracking_error_and_holding_checks() -> None:
    contract = _contract()
    assert contract.metrics_satisfy_result_gates(_metrics(annual_tracking_error=0.0))
    assert not contract.metrics_satisfy_result_gates(
        _metrics(notional_survival_at_30_sessions=0.19)
    )
    assert not contract.metrics_satisfy_result_gates(
        _metrics(fold_censored_notional_fraction=0.51)
    )


def test_rank_key_prioritizes_active_return_lower_confidence_bound() -> None:
    high_point_low_lcb = _metrics(
        net_active_return_20bp=0.10,
        block_bootstrap_lcb95_net_active_return_20bp=0.001,
    )
    lower_point_high_lcb = _metrics(
        net_active_return_20bp=0.03,
        block_bootstrap_lcb95_net_active_return_20bp=0.01,
    )
    assert lower_point_high_lcb.rank_key < high_point_low_lcb.rank_key


def test_public_receipt_factory_requires_real_receipt_and_arrays_then_stops() -> None:
    parameters = inspect.signature(
        build_m03r_verified_inner_evaluator_receipt
    ).parameters
    assert "metrics" not in parameters
    assert "chronological_evaluator_receipt" in parameters
    assert "policy_net_returns" in parameters

    receipt, arrays = _chronological_evidence()
    with pytest.raises(M03RSelectionError, match="adapter is unavailable"):
        build_m03r_verified_inner_evaluator_receipt(
            chronological_evaluator_receipt=receipt,
            **arrays,
        )


def test_chronological_receipt_is_recomputed_not_trusted_by_hash() -> None:
    receipt, arrays = _chronological_evidence()
    mutated_arrays = dict(arrays)
    mutated_policy = np.asarray(arrays["policy_net_returns"]).copy()
    mutated_policy[0, 0] += 1e-12
    mutated_arrays["policy_net_returns"] = mutated_policy
    with pytest.raises(M03RSelectionError, match="typed-array reproduction"):
        validate_m03r_chronological_evaluator_evidence(
            chronological_evaluator_receipt=receipt,
            **mutated_arrays,
        )


def test_inventory_and_cross_generation_identities_fail_closed() -> None:
    inventory = _inventory()
    with pytest.raises(M03RSelectionError, match="canonical fold/seed order"):
        _contract(expected_fold_seed_inventory=tuple(reversed(inventory)))
    with pytest.raises(M03RSelectionError, match="duplicate"):
        _contract(expected_fold_seed_inventory=(*inventory, inventory[-1]))
    with pytest.raises(M03RSelectionError, match="cannot identify"):
        _contract(protocol_generation=M03R_SUPERSEDED_PROTOCOL_GENERATION)
