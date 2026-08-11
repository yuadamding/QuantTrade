from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.training.top2000_m03r_v8_sentinel import (
    M03RV8DistinctPolicySentinelError,
    build_m03r_v8_synthetic_setting_cases,
    collapsed_m03r_v8_sentinel_fixture,
    run_m03r_v8_distinct_policy_sentinel,
)


def test_all_eight_causal_rows_remain_distinct_through_execution() -> None:
    result = run_m03r_v8_distinct_policy_sentinel()

    assert result.passed
    assert result.unique_causal_input_count == 5
    assert result.unique_gated_proposal_count == 7
    assert result.unique_projected_weights_count == 8
    assert result.unique_executed_weights_count == 8
    assert len({row.action_trace_receipt_sha256 for row in result.rows}) == 8
    assert result.cpu_only
    assert not result.gpu_capacity_evidence
    assert not result.performance_evidence
    assert len(result.receipt_sha256) == 64


def test_duplicate_causal_input_fails_the_sentinel_without_false_success() -> None:
    result = run_m03r_v8_distinct_policy_sentinel(collapsed_m03r_v8_sentinel_fixture())

    assert not result.passed
    assert result.unique_causal_input_count == 4
    assert result.unique_gated_proposal_count < 8
    assert result.unique_executed_weights_count < 8


def test_case_inventory_and_rehashed_status_tampering_fail_closed() -> None:
    cases = build_m03r_v8_synthetic_setting_cases()
    with pytest.raises(M03RV8DistinctPolicySentinelError, match="eight-setting"):
        run_m03r_v8_distinct_policy_sentinel(cases[:-1])

    result = run_m03r_v8_distinct_policy_sentinel()
    with pytest.raises(M03RV8DistinctPolicySentinelError, match="status"):
        replace(result, passed=False).validate()
    with pytest.raises(M03RV8DistinctPolicySentinelError, match="uniqueness"):
        replace(result, unique_executed_weights_count=7).validate()
